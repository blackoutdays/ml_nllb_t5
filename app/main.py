from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.translate import INPUT_CSV, process_csv
import os

app = FastAPI(
    title="CSV Translator API",
    description="API for translating products 1688 from en -> ru  (t5 translate model large 1024) :)))) ",
    version="1.0",
)

@app.post("/translate/", summary="Перевести CSV", description="Читает CSV из директории, переводит и сохраняет.")
async def start_translation():
    if not os.path.exists(INPUT_CSV):
        raise HTTPException(status_code=404, detail="Файл не найден")

    try:
        translated_file = await process_csv()
        return JSONResponse(content={"message": "Перевод завершен!", "file": translated_file})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла ошибка: {str(e)}")