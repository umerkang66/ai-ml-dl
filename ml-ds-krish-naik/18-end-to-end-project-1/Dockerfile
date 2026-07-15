FROM python:3.12-slim-bullseye

WORKDIR /app

COPY . /app

RUN apt update -y && apt install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "application.py"]
