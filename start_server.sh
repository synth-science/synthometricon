#!/usr/bin/env bash
# Launch llama-server from config.yaml (llama_cpp.server) under a restart loop,
# teeing output to logs/llama-server-<ts>.log. Keys: docs/configuration.md#llama_cpp.
# Usage: ./start_server.sh [config.yaml]
set -euo pipefail
CONFIG="${1:-config.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/llama-server-$(date +%Y%m%d_%H%M%S).log"

# Null-delimited argv so arguments with spaces survive.
mapfile -d '' ARGS < <(python3 - "$CONFIG" <<'PYEOF'
import os, sys, yaml
s = yaml.safe_load(open(sys.argv[1]))["llama_cpp"]["server"]

# LLAMA_PORT wins so co-located array tasks get unique ports.
port = os.environ.get("LLAMA_PORT") or s["port"]

args = [
    "llama-server",
    "--host", s["host"],
    "--port", str(port),
    "--ctx-size", str(s["ctx_size"]),
    "--batch-size", str(s["batch_size"]),
    "--ubatch-size", str(s["ubatch_size"]),
    "--image-min-tokens", str(s["image_min_tokens"]),
    "--image-max-tokens", str(s["image_max_tokens"]),
]

# on|off|auto; required for a quantized KV cache.
if s.get("flash_attn") is not None:
    fa = s["flash_attn"]
    args += ["--flash-attn", "on" if fa is True else "off" if fa is False else str(fa)]

# KV-cache quantization (e.g. q8_0 ~halves KV memory).
if s.get("cache_type_k"):
    args += ["--cache-type-k", str(s["cache_type_k"])]
if s.get("cache_type_v"):
    args += ["--cache-type-v", str(s["cache_type_v"])]

# Server slots; unset => llama.cpp auto (4).
if s.get("n_parallel"):
    args += ["--parallel", str(s["n_parallel"])]

# Prefer a pre-staged local GGUF (offline compute nodes) over -hf.
if s.get("model_path"):
    args += ["-m", s["model_path"]]
else:
    args += ["-hf", s["hf_repo"]]

# Vision projector: -hf auto-fetches it, a local -m does not.
if s.get("mmproj_path"):
    args += ["--mmproj", s["mmproj_path"]]

# Host-RAM prompt cache (MiB); 0 disables it (GB-scale saves can OOM the host).
if s.get("cache_ram") is not None:
    args += ["--cache-ram", str(s["cache_ram"])]

if s.get("spec_type"):
    args += ["--spec-type", s["spec_type"]]
# Draft model: local path wins over hf repo, as for the main model.
if s.get("spec_draft_path"):
    args += ["--model-draft", s["spec_draft_path"]]
elif s.get("spec_draft_hf_repo"):
    args += ["--spec-draft-hf", s["spec_draft_hf_repo"]]

sys.stdout.write("\0".join(args))
PYEOF
)

echo "llama-server logging to: $LOGFILE" >&2
echo "command: ${ARGS[*]}" | tee "$LOGFILE" >&2

# Restart on abnormal exit; stop on clean exit or Ctrl+C/SIGTERM.
RESTART_DELAY="${RESTART_DELAY:-2}"

# The signal also kills llama-server; the trap (deferred until the pipeline
# returns) just marks the shutdown as intentional.
SHOULD_RUN=1
trap 'SHOULD_RUN=0' INT TERM

while [[ "$SHOULD_RUN" -eq 1 ]]; do
  # set +e so a crash is inspected rather than aborting the script.
  set +e
  "${ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$SHOULD_RUN" -eq 0 ]]; then
    echo "shutdown requested; not restarting llama-server" | tee -a "$LOGFILE" >&2
    break
  fi
  if [[ "$status" -eq 0 ]]; then
    echo "llama-server exited cleanly (status 0); not restarting" | tee -a "$LOGFILE" >&2
    break
  fi
  # 130/143 = SIGINT/SIGTERM delivered only to the child: still intentional.
  if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
    echo "llama-server terminated by signal (status $status); not restarting" | tee -a "$LOGFILE" >&2
    break
  fi
  echo "llama-server crashed (status $status); restarting in ${RESTART_DELAY}s..." | tee -a "$LOGFILE" >&2
  sleep "$RESTART_DELAY"
done
