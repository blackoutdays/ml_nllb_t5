import os
import time
import torch
import json
import logging
import aiofiles
import asyncio
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
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
    torch.cuda.empty_cache()  # Очистка памяти перед запуском
else:
    logger.warning("GPU недоступен, используем CPU.")

logger.info("Загружаем модель...")
MODEL_PATH = "/models/t5_translate_model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
logger.info("Модель загружена!")

BATCH_SIZE = 150  # Количество товаров в одном батче

def get_dynamic_threads():
    """Определяет количество потоков в зависимости от загрузки GPU"""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 10  # Если GPU нет, используем минимальный лимит

        load = gpus[0].load  # Загруженность GPU (0.0 - 1.0)
        logger.info(f"🎛 Загруженность GPU: {load * 100:.2f}%")

        if load > 0.8:
            return 10  # Если GPU сильно загружен, уменьшаем до 10 потоков
        return 20  # Если GPU не загружен, используем 20 потоков
    except Exception as e:
        logger.warning(f"⚠ Ошибка при определении загрузки GPU: {e}")
        return 10  # В случае ошибки безопасно ставим 10 потоков

NUM_THREADS = get_dynamic_threads()
semaphore = asyncio.Semaphore(NUM_THREADS)  # Контролируем количество активных задач

async def write_to_csv(rows):
    """Асинхронно записывает несколько строк в CSV."""
    async with aiofiles.open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        await f.writelines([",".join(map(str, row)) + "\n" for row in rows])

def parse_json_safe(x):
    """Безопасный разбор JSON."""
    try:
        return json.loads(x) if isinstance(x, str) and x.startswith("{") else {}
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка JSON: {e} в строке: {x}")
        return {}

def translate_text(text):
    """Переводит ОДИН текст (работает в отдельном потоке) и логирует результат."""
    if not text or pd.isna(text):
        return ""

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=256,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.1,
            num_beams=5,
            early_stopping=True,
        )

    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    logger.info(f" Переведено: \"{text}\" → \"{translated_text}\"")
    return translated_text

async def translate_batch(batch):
    """Асинхронно переводит БАТЧ из 150 товаров."""
    logger.info(f"Переводим {len(batch)} товаров...")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        translations = await asyncio.gather(*[loop.run_in_executor(executor, translate_text, row["en"]) for _, row in batch.iterrows()])

    return translations

async def process_batch(batch):
    """Обрабатывает один батч товаров: переводит и записывает в CSV с логами перевода."""
    async with semaphore:  # Ограничиваем число активных задач
        start_time = time.time()
        translations = await translate_batch(batch)

        csv_rows = []
        for i, (_, row) in enumerate(batch.iterrows()):
            translated_text = translations[i]
            logger.info(f"Переведено ID {row['id']}: \"{row['en']}\" → \"{translated_text}\"")  # Лог перевода
            csv_rows.append([row["id"], row["en"], translated_text, row["product_id"], row["category_id"]])

        await write_to_csv(csv_rows)

        elapsed_time = time.time() - start_time
        logger.info(f"Обработано {len(batch)} строк за {elapsed_time:.2f} сек.")

async def process_csv():
    """Обрабатывает CSV, распределяя батчи на 10-20 потоков в зависимости от загрузки GPU."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"⚠ Файл не найден: {INPUT_CSV}")

    try:
        df = await asyncio.to_thread(pd.read_csv, INPUT_CSV, delimiter=";", quotechar='"', on_bad_lines="skip", dtype=str)
        logger.info(f" CSV загружен, строк: {len(df)}")
    except Exception as e:
        logger.error(f" Ошибка чтения CSV: {e}")
        return None

    if "id" not in df.columns or "names" not in df.columns:
        logger.error(" CSV не содержит необходимые столбцы 'id' и 'names'.")
        return None

    df.dropna(subset=["id", "names"], inplace=True)
    logger.info(f" Очищенный CSV: {len(df)} строк после удаления пустых значений.")

    df["names"] = df["names"].apply(parse_json_safe)
    df["en"] = df["names"].apply(lambda x: x.get("en", "") if isinstance(x, dict) else "")

    if not os.path.exists(OUTPUT_CSV):
        async with aiofiles.open(OUTPUT_CSV, mode="w", encoding="utf-8") as f:
            await f.write("id,en_name,ru_name,product_id,category_id\n")

    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    logger.info(f" Всего {len(batches)} батчей по {BATCH_SIZE} товаров.")

    await asyncio.gather(*(process_batch(batch) for batch in batches))

    logger.info(f" Перевод завершен! Файл сохранен: {OUTPUT_CSV}")
    return OUTPUT_CSV