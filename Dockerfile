FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y python3-pip && pip3 install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]