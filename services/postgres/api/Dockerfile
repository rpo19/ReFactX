FROM python:3.12

WORKDIR /app

COPY --from=repo . /app

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements/api.txt -r requirements/base.txt -r requirements/postgres.txt -r requirements/transformers.txt

EXPOSE 8000

ENV WORKERS_NUM=4

CMD ["sh", "-c", "gunicorn -w $WORKERS_NUM -b 0.0.0.0:8000 api:app"]
