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


async def write_to_csv(rows):
    """Асинхронно записывает несколько строк в CSV."""
    async with aiofiles.open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        await f.writelines([",".join(map(str, row)) + "\n" for row in rows])


def parse_json_safe(x):
    """Безопасный разбор JSON с логированием ошибок."""
    try:
        return json.loads(x) if isinstance(x, str) and x.startswith("{") else {}
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка JSON: {e} в строке: {x}")
        return {}


async def translate_batch(texts):
    """Перевод батча текстов с обработкой ошибок."""
    if not texts:
        logger.warning("Пустой список текстов для перевода!")
        return []

    logger.info(f"Отправка {len(texts)} текстов в модель...")

    try:
        async def generate():
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
            return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        translations = await asyncio.to_thread(generate)
        return translations

    except Exception as e:
        logger.error(f"Ошибка при переводе батча: {e}")
        return ["" for _ in texts]  # Возвращаем пустые строки, чтобы не ломался порядок


async def process_batch(rows, translated_ids):
    """Обрабатывает батч переводов с параллельной записью в CSV."""
    start_time = time.time()
    rows_to_translate = rows[~rows["id"].astype(str).isin(translated_ids)]

    if rows_to_translate.empty:
        logger.info("Все строки в этом батче уже переведены, пропускаем.")
        return

    texts = rows_to_translate["en"].tolist()
    translations = await translate_batch(texts)

    csv_rows = [
        [row["id"], row["en"], translations[i], row["product_id"], row["category_id"]]
        for i, (_, row) in enumerate(rows_to_translate.iterrows())
    ]

    await write_to_csv(csv_rows)

    elapsed_time = time.time() - start_time
    logger.info(f"Обработано {len(rows_to_translate)} строк за {elapsed_time:.2f} секунд.")


async def load_existing_translations():
    """Загружает уже переведенные товары в отдельном потоке, чтобы не блокировать выполнение."""
    if not os.path.exists(OUTPUT_CSV):
        return set()

    try:
        df_translated = await asyncio.to_thread(pd.read_csv, OUTPUT_CSV, delimiter=",", quotechar='"',
                                                on_bad_lines="skip")
        return set(df_translated["id"].astype(str))
    except Exception as e:
        logger.error(f"Ошибка загрузки переведенных данных: {e}")
        return set()


async def get_optimal_concurrency():
    """Динамически определяет количество асинхронных потоков на основе загрузки GPU."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 20

        load = gpus[0].load
        return int(20 + (40 - 20) * (1 - load)) if load < 0.9 else 20
    except Exception:
        return 20


async def get_dynamic_batch_size():
    """Динамически регулирует размер батча в зависимости от загрузки GPU."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 200  # Без GPU использовать стандартный размер

        load = gpus[0].load
        if load < 0.5:
            return 500  # Увеличиваем батч, если GPU недогружен
        elif load < 0.8:
            return 300  # Средняя загрузка, батч чуть больше
        return 200  # Высокая загрузка, стандартный размер
    except Exception as e:
        logger.warning(f"Ошибка при определении загрузки GPU: {e}")
        return 200  # По умолчанию


async def process_csv():
    """Обрабатывает CSV с ограничением количества параллельных задач через asyncio.Semaphore."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    try:
        df = await asyncio.to_thread(pd.read_csv, INPUT_CSV, on_bad_lines="skip", delimiter=";", quotechar='"')
    except Exception as e:
        logger.error(f"Ошибка чтения CSV: {e}")
        return None

    if "names" not in df.columns:
        raise ValueError("CSV не содержит столбца 'names'")

    df["names"] = df["names"].apply(parse_json_safe)
    df["en"] = df["names"].apply(lambda x: x.get("en", "") if isinstance(x, dict) else "")

    if not os.path.exists(OUTPUT_CSV):
        async with aiofiles.open(OUTPUT_CSV, mode="w", encoding="utf-8") as f:
            await f.write("id,en_name,ru_name,product_id,category_id\n")

    batch_size = await get_dynamic_batch_size()
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    translated_ids = await load_existing_translations()
    concurrency_limit = await get_optimal_concurrency()
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def process_limited(batch, batch_index):
        async with semaphore:
            await process_batch(batch, translated_ids)

    await asyncio.gather(*(process_limited(batch, i) for i, batch in enumerate(batches)))

    logger.info(f"Перевод завершен! Файл сохранен: {OUTPUT_CSV}")
    return OUTPUT_CSV