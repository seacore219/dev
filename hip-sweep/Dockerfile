# syntax=docker/dockerfile:1

FROM continuumio/miniconda2:4.7.12@sha256:c1c84747f3ed7c4eca1f59c2c2ea954565249853b791b0c8fbf9fc866b715f4f

ARG SORN_COMMIT=cdad55d55f39e04f568ca1bc0c6036bec8db08fb

LABEL org.opencontainers.image.source="https://github.com/delpapa/SORN" \
      org.opencontainers.image.revision="${SORN_COMMIT}" \
      org.opencontainers.image.description="Original Python 2 SORN, FrozenPlasticity h_ip sweep"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN conda install --yes \
        numpy=1.16.6 \
        matplotlib=2.2.3 \
        pytables=3.5.2 \
    && pip install --no-cache-dir --only-binary=:all: \
        scipy==1.2.3 \
    && pip install --no-cache-dir \
        ipdb==0.13.9 \
        powerlaw==1.4.6 \
    && conda clean --all --yes

RUN mkdir -p /opt/sorn \
    && wget --quiet --output-document=/tmp/sorn.tar.gz \
       "https://github.com/delpapa/SORN/archive/${SORN_COMMIT}.tar.gz" \
    && tar --extract --gzip --file=/tmp/sorn.tar.gz \
       --directory=/opt/sorn --strip-components=1 \
    && rm /tmp/sorn.tar.gz

RUN sed -i \
      "s/start_seed = current_time_int\*(rank + 1)/start_seed = int(os.environ.get('SORN_SEED', current_time_int*(rank + 1)))/" \
      /opt/sorn/utils/backup.py \
    && sed -i \
      's#utils.initialise_backup(mount="../", dest="../backup")#utils.initialise_backup(mount="../", dest=os.path.join("../backup", os.environ.get("SORN_RUN_ID", "manual")))#' \
      /opt/sorn/common/test_single.py \
    && sed -i '/from __future__ import division/a import os' /opt/sorn/delpapa/param_Zheng2013.py \
    && sed -i \
      "s/c.h_ip = 0.1/c.h_ip = float(os.environ.get('SORN_H_IP', 0.1))/" \
      /opt/sorn/delpapa/param_Zheng2013.py \
    && sed -i \
      "s/c.steps_plastic = 5000000/c.steps_plastic = 6000000/" \
      /opt/sorn/delpapa/param_Zheng2013.py \
    && grep -Fq "start_seed = int(os.environ.get('SORN_SEED'" /opt/sorn/utils/backup.py \
    && grep -Fq "c.h_ip = float(os.environ.get('SORN_H_IP'" /opt/sorn/delpapa/param_Zheng2013.py \
    && grep -Fq 'c.N_e = 200' /opt/sorn/delpapa/param_Zheng2013.py \
    && grep -Fq 'c.steps_plastic = 6000000' /opt/sorn/delpapa/param_Zheng2013.py

COPY entrypoint.sh /usr/local/bin/sorn-hipsweep
RUN chmod 0755 /usr/local/bin/sorn-hipsweep \
    && python -c "import numpy, scipy, tables, matplotlib, powerlaw; print('Python 2 scientific stack validated')"

WORKDIR /opt/sorn/common
VOLUME ["/opt/sorn/backup"]

ENTRYPOINT ["/usr/local/bin/sorn-hipsweep"]