import os
import time
import torch
import json
import logging
import aiofiles
import asyncio
import pandas as pd
from multiprocessing import cpu_count
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("translation.log"), logging.StreamHandler()],
)
logger = logging.getLogger()

# MODEL_PATH = "/Users/aruka/PycharmProjects/NLLP_model/models/t5_translate_model"
# INPUT_CSV = "/Users/aruka/PycharmProjects/NLLP_model/app/data/products.csv"
# OUTPUT_CSV = "/Users/aruka/PycharmProjects/NLLP_model/app/data/translated_products.csv"

MODEL_PATH = "/share/t5_translate_en_ru_zh_large_1024"
INPUT_CSV = "/home/aruzhan/products_202503121705.csv"
OUTPUT_CSV = "/home/aruzhan/translated_products.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("GPU недоступен, используем CPU.")

logger.info("Загружаем модель...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
logger.info("Модель загружена!")

NUM_THREADS = cpu_count()
logger.info(f"Количество потоков: {NUM_THREADS}")

async def write_to_csv(row):
    """Асинхронно записывает ОДНУ строку в CSV."""
    async with aiofiles.open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        await f.write(",".join(map(str, row)) + "\n")

def translate_text(text):
    if not text or pd.isna(text):
        return ""

    input_text = f"translate eng to rus: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

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

def parse_json_safe(x):
    try:
        return json.loads(x) if isinstance(x, str) and x.startswith("{") else {}
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка JSON: {e} для строки: {x}")
        return {}

def process_row(row):
    """Переводит одну строку и сразу записывает её в CSV."""
    start_time = time.time()
    logger.info(f"Переводим ID {row['id']}: {row['en']}")

    translated_text = translate_text(row['en'])

    elapsed_time = time.time() - start_time
    logger.info(f"Переведено: {translated_text} (⏳ {elapsed_time:.2f} сек)")

    return [row['id'], row['en'], translated_text, row['product_id'], row['category_id']]

async def process_csv():
    """Обрабатывает CSV по строкам, переводит и сразу записывает."""
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV, on_bad_lines="skip", delimiter=";", quotechar='"')
    except Exception as e:
        logger.error(f"Ошибка чтения CSV: {e}")
        return None

    logger.info(f"Колонки CSV: {df.columns}")

    if "names" not in df.columns:
        raise ValueError("CSV не содержит столбца 'names'")

    df["names"] = df["names"].apply(parse_json_safe)
    df["en"] = df["names"].apply(lambda x: x.get("en", "") if isinstance(x, dict) else "")

    logger.info("Начинаем перевод...")

    if not os.path.exists(OUTPUT_CSV):
        async with aiofiles.open(OUTPUT_CSV, mode="w", encoding="utf-8") as f:
            await f.write("id,en_name,ru_name,product_id,category_id\n")


    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        loop = asyncio.get_running_loop()
        for _, row in df.iterrows():
            translated_row = await loop.run_in_executor(executor, process_row, row)
            await write_to_csv(translated_row)

    logger.info(f"Перевод завершен! Файл сохранен: {OUTPUT_CSV}")
    return OUTPUT_CSV