#!/usr/bin/env bash
set -euo pipefail

: "${SORN_RUN_ID:?Set SORN_RUN_ID}"
: "${SORN_SEED:?Set SORN_SEED}"
: "${SORN_H_IP:?Set SORN_H_IP}"

case "$SORN_RUN_ID" in
  *[!A-Za-z0-9._/-]*)
    echo "SORN_RUN_ID may contain only letters, digits, dot, underscore, slash, and hyphen" >&2
    exit 64
    ;;
esac

case "$SORN_SEED" in
  ''|*[!0-9]*)
    echo "SORN_SEED must be a non-negative integer" >&2
    exit 64
    ;;
esac

mkdir -p "/opt/sorn/backup/$SORN_RUN_ID"
export SORN_RUN_ID SORN_SEED SORN_H_IP

echo "Starting original SORN commit cdad55d55f39e04f568ca1bc0c6036bec8db08fb"
echo "Experiment: delpapa.param_Zheng2013; N_e=200; steps=6000000; h_ip=$SORN_H_IP"
echo "Run ID: $SORN_RUN_ID; seed: $SORN_SEED"

exec /opt/conda/bin/python -u test_single.py delpapa.param_Zheng2013