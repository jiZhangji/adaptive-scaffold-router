#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-status}"
HEAD_IP="${HEAD_IP:-10.6.2.4}"
WORKER_IP="${WORKER_IP:-10.6.2.3}"
PEER_HOST="${PEER_HOST:-powerleader@10.6.2.3}"
PEER_KEY="${PEER_KEY:-$HOME/.ssh/a6000_peer_ed25519}"
ENV_PREFIX="${ENV_PREFIX:-/home/powerleader/project/envs/scaf-grpo}"
RAY_BIN="${RAY_BIN:-$ENV_PREFIX/bin/ray}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
RAY_PORT="${RAY_PORT:-6379}"

SSH=(ssh -i "$PEER_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)

remote() {
  "${SSH[@]}" "$PEER_HOST" "$@"
}

network_env=(
  NCCL_SOCKET_IFNAME=eno1
  GLOO_SOCKET_IFNAME=eno1
  NCCL_IB_DISABLE=1
  NCCL_DEBUG=WARN
  RAY_DISABLE_DOCKER_CPU_WARNING=1
)

case "$ACTION" in
  start)
    [[ -x "$RAY_BIN" ]] || { echo "Missing Ray binary: $RAY_BIN" >&2; exit 2; }
    [[ -s "$PEER_KEY" ]] || { echo "Missing peer key: $PEER_KEY" >&2; exit 2; }
    if pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo' >/dev/null; then
      echo "Refusing to reset Ray while a local training process is active." >&2
      exit 3
    fi
    if remote "pgrep -af '[v]erl.trainer.main_ppo|[h]int_mix_grpo.main_ppo' >/dev/null"; then
      echo "Refusing to reset Ray while a peer training process is active." >&2
      exit 3
    fi
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    remote "'$RAY_BIN' stop --force >/dev/null 2>&1 || true"
    env "${network_env[@]}" "$RAY_BIN" start --head \
      --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-gpus=1 \
      --node-manager-port=10002 --object-manager-port=10003 \
      --min-worker-port=11000 --max-worker-port=11999 \
      --dashboard-host=127.0.0.1 --disable-usage-stats >/tmp/td_ray_head.log
    remote "env ${network_env[*]} '$RAY_BIN' start \
      --address='$HEAD_IP:$RAY_PORT' --node-ip-address='$WORKER_IP' --num-gpus=1 \
      --node-manager-port=10002 --object-manager-port=10003 \
      --min-worker-port=11000 --max-worker-port=11999 \
      --disable-usage-stats >/tmp/td_ray_worker.log"
    RAY_ADDRESS=auto "$PYTHON_BIN" - <<'PY'
import time
import ray

ray.init(address="auto")
for _ in range(60):
    resources = ray.cluster_resources()
    if resources.get("GPU", 0) >= 2 and len(ray.nodes()) >= 2:
        print({"status": "ready", "resources": resources, "nodes": len(ray.nodes())})
        break
    time.sleep(2)
else:
    raise SystemExit(f"two-node cluster did not become ready: {ray.cluster_resources()}")
PY
    ;;
  stop)
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    remote "'$RAY_BIN' stop --force >/dev/null 2>&1 || true"
    echo "two-node Ray cluster stopped"
    ;;
  status)
    echo "===== HEAD $HEAD_IP ====="
    "$RAY_BIN" status || true
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader
    echo "===== WORKER $WORKER_IP ====="
    remote "hostname; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; '$RAY_BIN' status 2>/dev/null || true"
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
