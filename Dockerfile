FROM ghcr.io/timeeval/python3-torch:0.3.0

LABEL description="DeepAnT + RW-ML + AutoCEGAR"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy project
COPY deepant /app/deepant
COPY cegar   /app/cegar
COPY config.py       /app/
COPY train_rwml.py   /app/
COPY eval_msl.py     /app/
COPY manifest.json   /app/

CMD ["python", "-m", "deepant.algorithm", "--help"]
