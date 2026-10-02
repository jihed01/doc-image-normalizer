FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN pip install --no-cache-dir uv \
    && uv sync --frozen

ENV PYTHONPATH=/app/src

CMD ["python", "-c", "print('Doc Image Normalizer container ready')"]