FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
CMD ["python", "-c", "print('Llama Fine-tuning Container Ready')"]