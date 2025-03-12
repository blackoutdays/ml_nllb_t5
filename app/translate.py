import os
import time
import torch
import json
import logging
import aiofiles
import asyncio
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import GPUtil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("translation.log"), logging.StreamHandler()],
)
logger = logging.getLogger()

INPUT_CSV = "/home/aruzhan/products_202503121705.csv"
OUTPUT_CSV = "/home/aruzhan/translated_products.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("GPU недоступен, используем CPU.")

logger.info("Загружаем модель...")
MODEL_PATH = "/models/t5_translate_model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)  # Перенос модели на GPU
logger.info("Модель загружена!")

async def write_to_csv(row):
    """Асинхронно записывает ОДНУ строку в CSV."""
    async with aiofiles.open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        await f.write(",".join(map(str, row)) + "\n")

def translate_batch(texts):
    if not texts:
        logger.warning("Пустой список текстов для перевода!")
        return []

    logger.info(f"Отправка {len(texts)} текстов в модель...")

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.1,
            num_beams=7,
            early_stopping=True,
        )

    translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    logger.info(f"Переведено {len(translations)} текстов.")
    return translations


async def process_batch(rows, translated_ids):
    """Обрабатывает батч переводов, пропуская уже переведенные товары."""
    start_time = time.time()
    rows_to_translate = rows[~rows["id"].astype(str).isin(translated_ids)]

    if rows_to_translate.empty:
        logger.info("Все строки в этом батче уже переведены, пропускаем.")
        return

    texts = rows_to_translate["en"].tolist()
    translations = translate_batch(texts)

    for i, (_, row) in enumerate(rows_to_translate.iterrows()):
        translated_text = translations[i]
        logger.info(f" Переведено ID {row['id']}: {translated_text} ( {time.time() - start_time:.2f} сек)")
        await write_to_csv([row['id'], row['en'], translated_text, row['product_id'], row['category_id']])

    elapsed_time = time.time() - start_time
    logger.info(f"Обработано {len(rows_to_translate)} строк за {elapsed_time:.2f} секунд.")

async def load_existing_translations():
    """Загружает уже переведенные товары, чтобы не дублировать работу."""
    if not os.path.exists(OUTPUT_CSV):
        return set()

    try:
        async with aiofiles.open(OUTPUT_CSV, mode="r", encoding="utf-8") as f:
            lines = await f.readlines()
        if len(lines) < 2:
            return set()

        df_translated = pd.read_csv(OUTPUT_CSV, delimiter=",", quotechar='"', on_bad_lines="skip")
        return set(df_translated["id"].astype(str))
    except Exception as e:
        logger.error(f" Ошибка загрузки переведенных данных: {e}")
        return set()

def parse_json_safe(x):
    try:
        return json.loads(x) if isinstance(x, str) and x.startswith("{") else {}
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка JSON: {e} для строки: {x}")
        return {}


MIN_CONCURRENCY = 20
MAX_CONCURRENCY = 40
batch_size = 200

async def get_optimal_concurrency():
    """Динамически определяет оптимальное число потоков в зависимости от загрузки GPU."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            logger.info("Не удалось получить данные о загрузке GPU, используем минимальный лимит потоков.")
            return MIN_CONCURRENCY  # Если нет данных о GPU, ставим минимум

        load = gpus[0].load  # Загруженность GPU (0.0 - 1.0)
        logger.info(f"🎛 Загруженность GPU: {load * 100:.2f}%")
        if load > 0.9:
            return MIN_CONCURRENCY  # Сильно загружен → минимальный лимит
        elif load < 0.5:
            return MAX_CONCURRENCY  # GPU недогружен → максимальный лимит
        else:
            return int((MAX_CONCURRENCY - MIN_CONCURRENCY) * (1 - load) + MIN_CONCURRENCY)
    except Exception as e:
        logger.warning(f"⚠ Ошибка при получении загрузки GPU: {e}")
        return MIN_CONCURRENCY

async def process_csv():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f" Файл не найден: {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV, on_bad_lines="skip", delimiter=";", quotechar='"')
    except Exception as e:
        logger.error(f" Ошибка чтения CSV: {e}")
        return None

    logger.info(f" Колонки CSV: {df.columns}")

    if "names" not in df.columns:
        raise ValueError(" CSV не содержит столбца 'names'")

    df["names"] = df["names"].apply(parse_json_safe)
    df["en"] = df["names"].apply(lambda x: x.get("en", "") if isinstance(x, dict) else "")

    logger.info("Начинаем перевод...")

    if not os.path.exists(OUTPUT_CSV):
        async with aiofiles.open(OUTPUT_CSV, mode="w", encoding="utf-8") as f:
            await f.write("id,en_name,ru_name,product_id,category_id\n")

    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    logger.info(f"Всего {len(batches)} батчей для перевода.")

    translated_ids = await load_existing_translations()
    logger.info(f" Найдено {len(translated_ids)} уже переведенных товаров.")

    concurrency_limit = await get_optimal_concurrency()
    logger.info(f"Оптимальное количество потоков: {concurrency_limit}")

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def process_limited(batch, batch_index):
        async with semaphore:
            logger.info(f"Обрабатываем батч {batch_index+1}/{len(batches)} ({len(batch)} строк)")
            await process_batch(batch, translated_ids)

    tasks = [process_limited(batch, i) for i, batch in enumerate(batches)]
    await asyncio.gather(*tasks)

    logger.info(f"Перевод завершен! Файл сохранен: {OUTPUT_CSV}")
    return OUTPUT_CSV
