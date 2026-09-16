FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

ENV MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY criticality_tumbleweed/ ./criticality_tumbleweed/

RUN pip install --no-cache-dir -r criticality_tumbleweed/requirements.txt \
    && pip install --no-cache-dir ./criticality_tumbleweed

COPY analysis.py .

CMD ["python", "-u", "analysis.py"]

RUN pip install --no-cache-dir -r criticality_tumbleweed/requirements.txt \
    && pip install --no-cache-dir ./criticality_tumbleweed \
    && pip install --no-cache-dir h5py