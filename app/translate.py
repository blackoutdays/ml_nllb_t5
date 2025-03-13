import os
import time
import torch
import gc
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
    torch.cuda.empty_cache()
else:
    logger.warning("GPU недоступен, используем CPU.")

logger.info("Загружаем модель...")
MODEL_PATH = "/models/t5_translate_model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device).half()
logger.info("Модель загружена!")

def adjust_batch_size():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 140

    load = gpus[0].load
    logger.info(f"Загруженность GPU: {load * 100:.2f}%")

    if load > 0.9:
        return 32
    return 140

BATCH_SIZE = adjust_batch_size()

def get_dynamic_threads():
    """Динамическая настройка количества потоков в зависимости от загрузки GPU, но с максимальным значением 60."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 10

        load = gpus[0].load
        logger.info(f"Загруженность GPU: {load * 100:.2f}%")

        if load > 0.8:
            return min(60, 20)
        return min(60, 50)

    except Exception as e:
        logger.warning(f"Ошибка при определении загрузки GPU: {e}")
        return 10

NUM_THREADS = get_dynamic_threads()
semaphore = asyncio.Semaphore(NUM_THREADS)

async def write_to_csv(rows):
    async with aiofiles.open(OUTPUT_CSV, mode="a", encoding="utf-8") as f:
        await f.write("\n".join([",".join(map(str, row)) for row in rows]) + "\n")

def parse_json_safe(x):
    try:
        return json.loads(x) if isinstance(x, str) and x.startswith("{") else {}
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка JSON: {e} в строке: {x}")
        return {}

def translate_text(text):
    if not text or pd.isna(text):
        return ""

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    input_text = f"translate eng to rus: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=256,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.1,
            num_beams=6,
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

async def translate_batch(batch, semaphore):
    logger.info(f"Переводим {len(batch)} товаров...")

    loop = asyncio.get_running_loop()
    tasks = []

    for _, row in batch.iterrows():
        tasks.append(loop.create_task(translate_text_with_semaphore(row["en"], semaphore)))

    translations = await asyncio.gather(*tasks)

    return translations

async def translate_text_with_semaphore(text, semaphore):
    async with semaphore:
        return await asyncio.to_thread(translate_text, text)

async def process_batch(batch, existing_ids, semaphore):
    start_time = time.time()

    rows_to_translate = []

    for _, row in batch.iterrows():
        item_id = str(row["id"])

        if item_id in existing_ids:
            logger.info(f"Пропускаю товар с ID {item_id}, он уже переведен.")
        else:
            rows_to_translate.append(row)

    if rows_to_translate:
        try:
            translations = await translate_batch(pd.DataFrame(rows_to_translate), semaphore)
        except Exception as e:
            logger.error(f"Ошибка при переводе: {e}")
            return

        rows_to_write = []
        for i, row in enumerate(rows_to_translate):
            translated_text = translations[i]
            logger.info(f"Переведено ID {row['id']}: \"{row['en']}\" → \"{translated_text}\"")
            csv_row = [row["id"], row["en"], translated_text, row["product_id"], row["category_id"]]
            rows_to_write.append(csv_row)
            existing_ids.add(str(row["id"]))

        if rows_to_write:
            await write_to_csv(rows_to_write)

    elapsed_time = time.time() - start_time
    logger.info(f"Обработано {len(batch)} строк за {elapsed_time:.2f} сек.")

    del batch
    gc.collect()
    torch.cuda.empty_cache()

async def process_csv():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    start_id = "7918904"  # Начинаем обработку с этого ID

    # 1️⃣ Загружаем уже переведённые ID (но не удаляем старые данные!)
    existing_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_ids = set(pd.read_csv(OUTPUT_CSV, usecols=["id"], dtype=str)["id"])
            logger.info(f"Найдено уже переведённых товаров: {len(existing_ids)}")
        except Exception as e:
            logger.warning(f"Ошибка при чтении OUTPUT_CSV: {e}")

    # 2️⃣ Загружаем весь INPUT_CSV
    try:
        df = await asyncio.to_thread(pd.read_csv, INPUT_CSV, delimiter=";", quotechar='"', on_bad_lines="skip", dtype=str)
        logger.info(f"CSV загружен, строк: {len(df)}")
    except Exception as e:
        logger.error(f"Ошибка чтения INPUT_CSV: {e}")
        return None

    if "id" not in df.columns or "names" not in df.columns:
        logger.error("CSV не содержит необходимые столбцы 'id' и 'names'.")
        return None

    df.dropna(subset=["id", "names"], inplace=True)
    df["id"] = df["id"].astype(str)  # Убеждаемся, что ID - строки

    # 3️⃣ Оставляем только товары с ID >= 7918904 и которых нет в OUTPUT_CSV
    df = df[(df["id"] >= start_id) & (~df["id"].isin(existing_ids))]
    logger.info(f"Оставшиеся товары для перевода после фильтрации: {len(df)}")

    if len(df) == 0:
        logger.info("Все нужные товары уже переведены. Завершаем работу.")
        return OUTPUT_CSV

    # 4️⃣ Приводим к нужному формату
    df.loc[:, "names"] = df["names"].apply(parse_json_safe)
    df.loc[:, "en"] = df["names"].apply(lambda x: x.get("en", "") if isinstance(x, dict) else "")

    # 5️⃣ Батчим только новые товары
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    logger.info(f"Всего {len(batches)} батчей по {BATCH_SIZE} товаров.")

    # 6️⃣ Переводим только новые товары и дописываем в OUTPUT_CSV
    await asyncio.gather(*(process_batch(batch, existing_ids, semaphore) for batch in batches))

    logger.info(f"Перевод завершён! Файл сохранён: {OUTPUT_CSV}")
    return OUTPUT_CSV