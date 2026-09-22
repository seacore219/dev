"""
Runs the pristine-repo FrozenPlasticity h_ip sweep as NRP Jobs: one pod per
run, 50 runs per h_ip value, each h_ip value's runs landing in their own
folder so analysis.py can pool same-h_ip runs together and never mix
different h_ip values.

Folder layout this produces, matching what analysis.py's BATCH_GLOB="h_ip_*"
and RUN_GLOB="run-*" expect:
  backup/h_ip_0.02/run-001/test_single/<timestamp>/common/result.h5
  backup/h_ip_0.02/run-002/test_single/<timestamp>/common/result.h5
  ...
  backup/h_ip_0.20/run-050/test_single/<timestamp>/common/result.h5

Run this from the repo root: python3 nrp-hip-sweep-launcher.py
"""

import json
import os
import subprocess
import time

NAMESPACE = 'hengenlab'
IMAGE = 'seacore219/sorn-hipsweep-py2:cdad55d'
PVC_NAME = 'aw222-sorn-replication-storage'
MOUNT_PATH = '/opt/sorn/backup'
JOB_PREFIX = 'charlesd-scalingline-hipsweep-'

# H_IP_VALUES = [round(0.02 + 0.02 * i, 2) for i in range(10)]  # 0.02 ... 0.20
H_IP_VALUES = [0.01, 0.03, 0.04, 0.06, 0.07] # bigger test
# H_IP_VALUES = [0.02] # test
N_RUNS_PER_HIP = 50

# One label identifying the whole sweep -- date, value range, and step size --
# used as the top-level folder every h_ip batch nests under.
import datetime
SWEEP_LABEL = '%s_hip%.2f-%.2f_step%.2f' % (
    datetime.datetime.now().strftime('%Y-%m-%d'),
    min(H_IP_VALUES), max(H_IP_VALUES),
    round(H_IP_VALUES[1] - H_IP_VALUES[0], 2) if len(H_IP_VALUES) > 1 else 0
)

CPU = "1"
MEMORY = "1Gi"
EPHEMERAL_STORAGE = "4Gi"

MAX_CONCURRENT = 50
POLL_SECONDS = 60

JOB_DIR = 'nrp_jobs'
if not os.path.exists(JOB_DIR):
    os.makedirs(JOB_DIR)

JOB_TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: sorn
          image: {image}
          imagePullPolicy: Always
          env:
            - name: SORN_RUN_ID
              value: "{run_id}"
            - name: SORN_SEED
              value: "{seed}"
            - name: SORN_H_IP
              value: "{h_ip}"
          resources:
            requests:
              cpu: "{cpu}"
              memory: "{memory}"
              ephemeral-storage: "{ephemeral_storage}"
            limits:
              cpu: "{cpu}"
              memory: "{memory}"
              ephemeral-storage: "{ephemeral_storage}"
          volumeMounts:
            - name: storage
              mountPath: {mount_path}
      volumes:
        - name: storage
          persistentVolumeClaim:
            claimName: {pvc_name}
"""


def build_run_list():
    runs = []
    for hip_index, h_ip in enumerate(H_IP_VALUES):
        for run_number in range(1, N_RUNS_PER_HIP + 1):
            run_id = '%s/h_ip_%.2f/run-%03d' % (SWEEP_LABEL, h_ip, run_number)
            # unique across both h_ip and run number, so no two jobs
            # anywhere in the whole sweep ever share a seed
            seed = 300000 + (hip_index * 1000) + run_number
            job_name = '%ship%02d-run%03d' % (JOB_PREFIX, hip_index, run_number)
            runs.append({'job_name': job_name, 'run_id': run_id, 'seed': seed, 'h_ip': h_ip})
    return runs


def count_active_jobs():
    out = subprocess.check_output(['kubectl', 'get', 'jobs', '-n', NAMESPACE, '-o', 'json'])
    data = json.loads(out)
    active = 0
    for item in data.get('items', []):
        if not item['metadata']['name'].startswith(JOB_PREFIX):
            continue
        status = item.get('status', {})
        if status.get('succeeded', 0) < 1 and status.get('failed', 0) < 1:
            active += 1
    return active


def submit_run(run):
    yaml_content = JOB_TEMPLATE.format(
        job_name=run['job_name'],
        namespace=NAMESPACE,
        image=IMAGE,
        run_id=run['run_id'],
        seed=run['seed'],
        h_ip=run['h_ip'],
        cpu=CPU,
        memory=MEMORY,
        ephemeral_storage=EPHEMERAL_STORAGE,
        mount_path=MOUNT_PATH,
        pvc_name=PVC_NAME
    )
    yaml_path = os.path.join(JOB_DIR, run['job_name'] + '.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    subprocess.call(['kubectl', 'delete', '-f', yaml_path, '--ignore-not-found=true'])
    result = subprocess.call(['kubectl', 'apply', '-f', yaml_path])
    if result == 0:
        print('Submitted %s (h_ip=%s, run_id=%s, seed=%s)' %
              (run['job_name'], run['h_ip'], run['run_id'], run['seed']))
    else:
        print('FAILED to submit %s -- kubectl exit code %d' % (run['job_name'], result))


def main():
    pending = build_run_list()
    print('Total runs: %d (%d h_ip values x %d runs each)' %
          (len(pending), len(H_IP_VALUES), N_RUNS_PER_HIP))

    while pending:
        active = count_active_jobs()
        slots = MAX_CONCURRENT - active
        if slots > 0:
            batch = pending[:slots]
            pending = pending[slots:]
            for run in batch:
                submit_run(run)
            print('%d runs remaining, ~%d now active' % (len(pending), active + len(batch)))
        else:
            print('At concurrency limit (%d active) -- waiting...' % active)
        time.sleep(POLL_SECONDS)

    print('All runs submitted. Check with: kubectl get jobs -n %s | grep %s' % (NAMESPACE, JOB_PREFIX))


if __name__ == '__main__':
    main()