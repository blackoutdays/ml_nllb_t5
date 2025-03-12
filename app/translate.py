import os
import time
import torch
import json
import logging
import aiofiles
import asyncio
import pandas as pd
import tracemalloc
from transformers import T5Tokenizer, T5ForConditionalGeneration
import GPUtil
from concurrent.futures import ThreadPoolExecutor

tracemalloc.start()

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
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
logger.info("Модель загружена!")

NUM_THREADS = 30  # 30 потоков
BATCH_SIZE = 150  # Обрабатываем по 150 товаров в потоке

async def write_to_csv(rows):
    """Асинхронно записывает несколько строк в CSV."""
    async with aiofiles.open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        await f.writelines([",".join(map(str, row)) + "\n" for row in rows])

def parse_json_safe(x):
    """Безопасный разбор JSON."""
    try:
        return json.loads(x) if isinstance(x, str) and x.startswith("{") else {}
    except json.JSONDecodeError as e:
        logger.error(f" Ошибка JSON: {e} в строке: {x}")
        return {}

def translate_text(text):
    """Переводит ОДИН текст (работает в отдельном потоке)."""
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

    return tokenizer.decode(output[0], skip_special_tokens=True)

async def translate_batch(batch):
    """Асинхронно переводит БАТЧ из 150 товаров."""
    logger.info(f" Переводим {len(batch)} товаров...")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        translations = await asyncio.gather(*[loop.run_in_executor(executor, translate_text, row["en"]) for _, row in batch.iterrows()])

    return translations

async def process_batch(batch):
    """Обрабатывает батч товаров: переводит и записывает в CSV."""
    start_time = time.time()

    translations = await translate_batch(batch)

    csv_rows = [
        [row["id"], row["en"], translations[i], row["product_id"], row["category_id"]]
        for i, (_, row) in enumerate(batch.iterrows())
    ]

    await write_to_csv(csv_rows)

    elapsed_time = time.time() - start_time
    logger.info(f" Обработано {len(batch)} строк за {elapsed_time:.2f} сек.")

async def process_csv():
    """Обрабатывает CSV, распределяя батчи на 30 потоков."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f" Файл не найден: {INPUT_CSV}")

    try:
        df = await asyncio.to_thread(pd.read_csv, INPUT_CSV, on_bad_lines="skip", delimiter=";", quotechar='"')
    except Exception as e:
        logger.error(f" Ошибка чтения CSV: {e}")
        return None

    if "names" not in df.columns:
        raise ValueError("⚠ CSV не содержит столбца 'names'")

    df["names"] = df["names"].apply(parse_json_safe)
    df["en"] = df["names"].apply(lambda x: x.get("en", "") if isinstance(x, dict) else "")

    if not os.path.exists(OUTPUT_CSV):
        async with aiofiles.open(OUTPUT_CSV, mode="w", encoding="utf-8") as f:
            await f.write("id,en_name,ru_name,product_id,category_id\n")

    # Разбиваем CSV на батчи по 150 товаров
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

    logger.info(f" Всего {len(batches)} батчей по {BATCH_SIZE} товаров.")

    # Асинхронная обработка батчей
    await asyncio.gather(*(process_batch(batch) for batch in batches))

    logger.info(f" Перевод завершен! Файл сохранен: {OUTPUT_CSV}")
    return OUTPUT_CSV
