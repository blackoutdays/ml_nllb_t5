import os
import time
import torch
import json
import logging
import aiofiles
import asyncio
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

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
        return []

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Перенос данных на GPU

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.1,
            num_beams=5,
            early_stopping=True,
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

async def process_batch(rows):
    start_time = time.time()
    texts = [row['en'] for _, row in rows.iterrows()]
    translations = translate_batch(texts)

    for i, row in rows.iterrows():
        translated_text = translations[i]
        logger.info(f"Переведено ID {row['id']}: {translated_text} ( {time.time() - start_time:.2f} сек)")
        await write_to_csv([row['id'], row['en'], translated_text, row['product_id'], row['category_id']])

def parse_json_safe(x):
    try:
        return json.loads(x) if isinstance(x, str) and x.startswith("{") else {}
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка JSON: {e} для строки: {x}")
        return {}

async def process_csv():
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

    batch_size = 100
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    # Вместо ThreadPoolExecutor → асинхронный вызов
    tasks = [process_batch(batch) for batch in batches]
    await asyncio.gather(*tasks)

    logger.info(f"Перевод завершен! Файл сохранен: {OUTPUT_CSV}")
    return OUTPUT_CSV