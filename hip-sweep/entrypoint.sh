#!/usr/bin/env bash
set -euo pipefail

: "${JOB_COMPLETION_INDEX:?This image requires an Indexed Job (completionMode: Indexed), which sets JOB_COMPLETION_INDEX}"
case "$JOB_COMPLETION_INDEX" in
  ''|*[!0-9]*)
    echo "JOB_COMPLETION_INDEX must be a non-negative integer" >&2
    exit 64
    ;;
esac

: "${RUNS_PER_HIP:?Set RUNS_PER_HIP}"
: "${HIP_VALUES:?Set HIP_VALUES as a comma-separated list}"
: "${SWEEP_LABEL:?Set SWEEP_LABEL}"

IFS=',' read -ra HIP_ARRAY <<< "$HIP_VALUES"

hip_index=$((JOB_COMPLETION_INDEX / RUNS_PER_HIP))
run_number=$((JOB_COMPLETION_INDEX % RUNS_PER_HIP + 1))

SORN_H_IP="${HIP_ARRAY[$hip_index]}"
SORN_RUN_ID=$(printf '%s/h_ip_%s/run-%03d' "$SWEEP_LABEL" "$SORN_H_IP" "$run_number")
# must match whatever generated the sweep's YAML -- currently 400000 + hip_index*1000 + run_number
SORN_SEED=$((400000 + hip_index * 1000 + run_number))

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