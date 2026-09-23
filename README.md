# Running sorn-hipsweep on a Kubernetes cluster

## 1. Build and push the image

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --tag <registry>/<image-name>:<tag> \
  --push \
  hip-sweep/
```

`<registry>` = Docker Hub username (`docker.io/` implied) or any registry
the target cluster's nodes can reach — private/air-gapped clusters need
their own registry, not Docker Hub. `--platform linux/amd64` matters if
you're building on Apple Silicon; cluster nodes are x86_64.

## 2. Point the Job YAML at that image + set params

In `nrp-hipsweep-sweep-job.yaml`:
- `spec.template.spec.containers[].image` = the tag you just pushed
- `metadata.namespace`, `volumes[].persistentVolumeClaim.claimName` = target
  cluster's namespace/PVC (must already exist there)
- `spec.completions` = `len(HIP_VALUES) * RUNS_PER_HIP` (!! must be exact !! 
  mismatches silently drop trailing h_ip values)
- `spec.parallelism` ≤ that namespace's pod quota
  (`kubectl describe resourcequota -n <namespace>`)
- `env: HIP_VALUES`, `RUNS_PER_HIP`, `SWEEP_LABEL`

`completionMode: Indexed` must stay set — `entrypoint.sh` requires
`JOB_COMPLETION_INDEX`, which only exists under Indexed mode. It's mapped
to h_ip/run via `hip_index = JOB_COMPLETION_INDEX / RUNS_PER_HIP`,
`run_number = JOB_COMPLETION_INDEX % RUNS_PER_HIP + 1`.

## 3. Apply

```bash
kubectl delete -f nrp-hipsweep-sweep-job.yaml --ignore-not-found=true
kubectl apply -f nrp-hipsweep-sweep-job.yaml
```

`delete` first makes this idempotent — safe to rerun on a partial prior
attempt. `kubeconfig` just needs to point at the target cluster; nothing
else in this flow changes per-cluster besides what's listed in step 2.

## 4. Monitor

```bash
kubectl get pods -n <namespace> -l job-name=sorn-hipsweep-sweep \
  --sort-by=.metadata.creationTimestamp
```
of couse many other commands exist that can list either real-time file 
saving/creation in the PVC, or even live memory and CPU count. See
https://kubernetes.io/docs/reference/kubectl/quick-reference/ for more