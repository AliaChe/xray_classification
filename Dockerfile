FROM python:3.10-slim

WORKDIR /app

COPY requirements/ requirements/

RUN pip install --no-cache-dir -r requirements/prod.txt

COPY app/ app/
COPY src/ src/
COPY configs/ configs/

RUN mkdir -p tmp

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]