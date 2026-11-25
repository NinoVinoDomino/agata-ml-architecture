FROM python:3.12-slim AS app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock \
    && groupadd -g 10001 agata \
    && useradd -M -u 10001 -g agata agata \
    && install -d -o agata -g agata /app/data/daily
COPY agata ./agata
USER agata
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "agata.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM public.ecr.aws/lambda/python:3.12 AS lambda
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock --target "${LAMBDA_TASK_ROOT}"
COPY agata "${LAMBDA_TASK_ROOT}/agata"
CMD ["agata.lambda_handler.handler"]
