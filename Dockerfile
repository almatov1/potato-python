FROM python:3.11-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN rm -rf /app/dataset

# Открываем порт, на котором будет работать Uvicorn (обычно 8000)
EXPOSE 8000

# Определяем команду для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Предполагается, что ваш основной файл приложения называется main.py,
# а экземпляр FastAPI приложения называется app.