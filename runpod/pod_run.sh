#!/bin/bash
# Inside the pod: install, fetch the open Italian audio, build synthetic
# stereo, tokenize, run the three stages, sample, evaluate, then wait to be
# pulled. --local-slice does the same on a couple of minutes of audio and
# a few CPU steps, and it never calls RunPod.
#
#   bash runpod/pod_run.sh --local-slice
#   bash runpod/pod_run.sh --full
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODE=local
case "${1:-}" in
  --full) MODE=full ;;
  --local-slice) MODE=local ;;
  *)
    echo "usage: bash runpod/pod_run.sh --full | --local-slice"
    exit 2
    ;;
esac

if [[ "$MODE" == local ]]; then
  WORK="$ROOT/runpod/local-out/work"
  OUT="$ROOT/runpod/local-out/report"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
  # An empty value hides any GPU so the smoke test stays on CPU.
  export CUDA_VISIBLE_DEVICES=""
else
  WORK=/workspace/work
  OUT="${PILOT_OUT:-/workspace/pilot-out}"
fi
mkdir -p "$WORK" "$OUT/logs" "$OUT/samples" "$OUT/stores" "$OUT/checkpoints"
# A PULLED file from an earlier ack must not skip the grace period.
rm -f "$OUT/PULLED"
exec > >(while IFS= read -r line; do printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$line"; done | tee -a "$OUT/logs/pod_run.log") 2>&1

status() { printf '%s\n' "$1" | tee "$OUT/status.txt"; }
resources() {
  echo "-- disk --"
  df -h . / /workspace 2>/dev/null || df -h .
  echo "-- memory --"
  free -g || true
}

finish() {
  local rc=$?
  echo "$rc" > "$OUT/exit_code"
  {
    echo "exit $rc"
    echo "-- last log lines --"
    tail -n 200 "$OUT/logs/pod_run.log" 2>/dev/null || true
  } > "$OUT/status.txt"
  resources || true
  if [[ "$MODE" == full ]]; then
    local grace="${PILOT_PULL_GRACE_S:-600}"
    if [[ "$rc" -ne 0 ]]; then
      rm -f "$OUT/PULLED"
      date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/FAILED"
      echo "failure: keeping the pod up for ${grace}s so the error can be pulled"
      sleep "$grace"
    else
      rm -f "$OUT/PULLED"
      date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/DONE"
      local i=0
      while [[ "$i" -lt "$grace" ]]; do
        if [[ -f "$OUT/PULLED" && "$OUT/PULLED" -nt "$OUT/DONE" ]]; then
          echo "pull acknowledged"
          break
        fi
        sleep 5
        i=$((i + 5))
      done
    fi
    if [[ -f "$ROOT/runpod/watchdog.sh" ]]; then
      bash "$ROOT/runpod/watchdog.sh" --once || true
    elif [[ -f /tmp/pilot-watchdog.sh ]]; then
      bash /tmp/pilot-watchdog.sh --once || true
    fi
  fi
  exit "$rc"
}
trap finish EXIT

status "start mode=$MODE"
echo "repo $ROOT"
resources || true
if [[ "$MODE" == full ]]; then
  nvidia-smi || true
  if [[ ! -f /tmp/pilot-watchdog.pid ]]; then
    setsid bash "$ROOT/runpod/watchdog.sh" </dev/null >>"$OUT/logs/watchdog.log" 2>&1 &
    echo $! > /tmp/pilot-watchdog.pid
  fi
  if [[ -n "${PILOT_PULL_TOKEN:-}" ]]; then
    PILOT_OUT="$OUT" PILOT_PULL_PORT="${PILOT_PULL_PORT:-8080}" \
      python3 "$ROOT/runpod/pull_server.py" >>"$OUT/logs/pull_server.log" 2>&1 &
    echo $! > /tmp/pull-server.pid
  fi
fi

PY=python3
install_local() {
  local venv="$ROOT/runpod/.venv-cpu"
  if [[ ! -x "$venv/bin/python" ]]; then
    rm -rf "$venv"
    if ! python3 -m venv "$venv"; then
      if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
      fi
      uv venv --python 3.12 "$venv"
    fi
  fi
  PY="$venv/bin/python"
  "$PY" -m pip install --disable-pip-version-check -q numpy soundfile soxr
}

install_torch_cpu() {
  "$PY" -m pip install --disable-pip-version-check -q \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip install --disable-pip-version-check -q \
    'transformers==4.46.3' safetensors huggingface_hub
}

install_pod() {
  local venv=/workspace/venv
  python3 -m venv --system-site-packages "$venv"
  PY="$venv/bin/python"
  if ! "$PY" -c 'import torch; assert torch.__version__.startswith("2.9") and torch.cuda.is_available()'; then
    echo "image torch is not 2.9+CUDA; installing the setup.sh pin"
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    uv pip install --python "$PY" torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$PY" triton || true
  fi
  "$PY" -m pip install --disable-pip-version-check -q \
    numpy 'transformers==4.46.3' safetensors soundfile huggingface_hub soxr
}

if [[ "$MODE" == local ]]; then
  status "install cpu venv"
  install_local
  "$PY" "$ROOT/runpod/timing_score.py" --self-check
else
  status "install pod env"
  install_pod
fi

MAX_AUDIO=0
MAX_BYTES=0
DIALOGUE_S=90
if [[ "$MODE" == local ]]; then
  MAX_AUDIO=40
  MAX_BYTES=250000000
  DIALOGUE_S=30
fi

export PYTHONPATH="$ROOT/runpod${PYTHONPATH:+:$PYTHONPATH}"
SKIP_PREP=0
if [[ "$MODE" == full && "${PILOT_RESUME:-}" == "1" ]]; then
  echo "waiting up to 20 min for uploaded stores"
  waited=0
  while [[ ! -f "$OUT/restore/READY" && "$waited" -lt 1200 ]]; do
    sleep 5
    waited=$((waited + 5))
  done
  if [[ -d "$OUT/restore/stores" ]]; then
    mkdir -p "$WORK"
    cp -a "$OUT/restore/stores" "$WORK/stores"
    if [[ -d "$OUT/restore/checkpoints" ]]; then
      mkdir -p "$WORK/runs"
      cp -a "$OUT/restore/checkpoints/." "$WORK/runs/"
    fi
    SKIP_PREP=1
    echo "restored stores from upload"
  fi
fi

if [[ "$SKIP_PREP" == 0 ]]; then
status "download open audio"
DL_ARGS=(--out "$WORK/raw" --max-bytes "$MAX_BYTES")
if [[ "$MODE" == full ]]; then
  MONO_H=$("$PY" -c 'import pilot_config; print(pilot_config.MONO_HOURS)')
  DL_ARGS+=(--max-mono-hours "$MONO_H")
else
  DL_ARGS+=(--max-audio-s "$MAX_AUDIO")
fi
"$PY" "$ROOT/runpod/download_open.py" "${DL_ARGS[@]}"
cp -f "$WORK/raw/ATTRIBUTION.txt" "$OUT/ATTRIBUTION.txt" || true
resources || true

status "synthetic stereo"
SYN_ARGS=(--manifest "$WORK/raw/manifest.jsonl" --work "$WORK/audio" --dialogue-s "$DIALOGUE_S")
if [[ "$MODE" == full ]]; then
  SYN_H=$("$PY" -c 'import pilot_config; print(pilot_config.SYNTH_STEREO_HOURS)')
  SYN_ARGS+=(--max-synth-hours "$SYN_H")
fi
"$PY" "$ROOT/runpod/prepare_audio.py" "${SYN_ARGS[@]}"
cp -f "$WORK/audio/plan.json" "$OUT/plan.json"
if [[ "$MODE" == full ]]; then
  rm -rf "$WORK/raw/mls" "$WORK/raw/fleurs"
  echo "deleted raw MLS and FLEURS after the dialogues were built"
fi
resources || true
fi

publish() {
  mkdir -p "$OUT/stores" "$OUT/checkpoints"
  if [[ -d "$WORK/stores" ]]; then
    mkdir -p "$OUT/stores"
    cp -a "$WORK/stores/." "$OUT/stores/"
  fi
  local s
  for s in s1 s2 s3; do
    if [[ -f "$WORK/runs/$s/ckpt.pt" ]]; then
      mkdir -p "$OUT/checkpoints/$s"
      cp -f "$WORK/runs/$s/ckpt.pt" "$OUT/checkpoints/$s/ckpt.pt"
    fi
    if [[ -f "$WORK/runs/$s/best.pt" ]]; then
      mkdir -p "$OUT/checkpoints/$s"
      cp -f "$WORK/runs/$s/best.pt" "$OUT/checkpoints/$s/best.pt"
    fi
  done
}

tokenize_one() {
  local clips="$1" dest="$2" width="$3"
  local extra=()
  if [[ "$MODE" == full && -n "${PILOT_BILLING_EPOCH:-}" ]]; then
    extra+=(--deadline-epoch "$PILOT_BILLING_EPOCH" --reserve-s 1200)
  fi
  if ! "$PY" "$ROOT/runpod/tokenize_store.py" \
      --clips "$clips" --out "$dest" --width "$width" "${extra[@]}"; then
    if [[ "$MODE" == local ]]; then
      echo "Mimi encode failed; writing placeholder tokens so train.py can still be exercised"
      "$PY" "$ROOT/runpod/tokenize_store.py" --clips "$clips" --out "$dest" --width "$width" --placeholder
    else
      return 1
    fi
  fi
  cp -f "$dest/speed.json" "$OUT/speed_w${width}.json" || true
}

status "tokenize"
# Torch is required for real codes. Install it on the local slice now,
# after the cheap audio steps, so a download bug fails before the big wheel.
if [[ "$MODE" == local ]]; then
  if ! "$PY" -c 'import torch' 2>/dev/null; then
    status "install cpu torch"
    install_torch_cpu
  fi
fi
"$PY" -c 'import numpy,soundfile,torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
if [[ "$MODE" == local ]]; then
  "$PY" "$ROOT/runpod/tokenize_store.py" --self-test
fi
mkdir -p "$WORK/stores"
if [[ "$SKIP_PREP" == 1 && -f "$WORK/stores/mono/tokens.bin" ]]; then
  echo "skipping tokenize; restored stores are present"
else
tokenize_one "$WORK/audio/mono.jsonl" "$WORK/stores/mono" 8
tokenize_one "$WORK/audio/stereo.jsonl" "$WORK/stores/stereo" 16
tokenize_one "$WORK/audio/dev.jsonl" "$WORK/stores/dev" 16
fi
publish || true
resources || true
export PILOT_OUT="$OUT"
export PILOT_WORK="$WORK"

if [[ "$MODE" == full ]]; then
  for store in mono stereo dev; do
    if grep -q placeholder_tokens "$WORK/stores/$store/meta.json"; then
      echo "refusing to train the pilot on placeholder tokens ($store)"
      exit 1
    fi
  done
fi

status "train"
export PY
export PYTHONPATH="$ROOT/runpod${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - <<PY
import os, sys
from pathlib import Path
sys.path.insert(0, "$ROOT/runpod")
import stages
mono = Path("$WORK/stores/mono")
stereo = Path("$WORK/stores/stereo")
dev = Path("$WORK/stores/dev")
runs = Path("$WORK/runs")
py = os.environ.get("PY", sys.executable)
if "$MODE" == "local":
    cmds = stages.cpu_smoke_commands(py, mono, stereo, dev, runs)
else:
    cmds = stages.full_commands(py, mono, stereo, dev, runs)
    stages.assert_full_numbers(cmds)
stages.run_commands(cmds, Path("$ROOT/third_party/caramel"), Path("$OUT/logs"))
PY
publish || true
resources || true

status "export"
src=""
if [[ -f "$WORK/runs/s3/best.pt" ]]; then
  src="$WORK/runs/s3/best.pt"
  cp -f "$src" "$OUT/best.pt"
elif [[ -f "$WORK/runs/s3/ckpt.pt" ]]; then
  src="$WORK/runs/s3/ckpt.pt"
elif [[ -f "$WORK/runs/s2/ckpt.pt" ]]; then
  src="$WORK/runs/s2/ckpt.pt"
elif [[ -f "$WORK/runs/s1/ckpt.pt" ]]; then
  src="$WORK/runs/s1/ckpt.pt"
fi
if [[ -z "$src" ]]; then
  echo "no checkpoint was written"
  exit 1
fi
echo "checkpoint $src"
(
  cd "$ROOT/third_party/caramel"
  "$PY" model.py "$src" "$OUT/caramel-it"
)

prompt=""
first_mono=$(find "$WORK/audio/mono" -name '*.flac' -o -name '*.wav' | head -n 1 || true)
if [[ -n "${first_mono:-}" ]] && command -v ffmpeg >/dev/null 2>&1; then
  prompt="$WORK/prompt.wav"
  ffmpeg -y -v error -i "$first_mono" -t 2 -ac 1 "$prompt" || prompt=""
fi

status "generate"
gen_one() {
  local seed="$1" seconds="$2"
  local out="$OUT/samples/it_${seed}.wav"
  local -a args=(--model "$OUT/caramel-it" --seconds "$seconds" --seed "$seed" --realtime --out "$out")
  if [[ -n "$prompt" ]]; then
    if (cd "$ROOT/third_party/caramel" && "$PY" generate.py --prompt "$prompt" "${args[@]}"); then
      return 0
    fi
    echo "prompted generate failed; retrying unprompted"
  fi
  (cd "$ROOT/third_party/caramel" && "$PY" generate.py "${args[@]}")
}

if [[ "$MODE" == local ]]; then
  if ! gen_one 0 1; then
    echo "CPU generate did not finish; training logs are still in $OUT/logs"
  fi
else
  "$PY" -m pip install --disable-pip-version-check -q faster-whisper || echo "faster-whisper install failed; ASR will be skipped"
  for seed in 0 1 2 3; do
    gen_one "$seed" 8
  done
fi

status "eval"
shopt -s nullglob
samples=()
for f in "$OUT/samples"/it_*.wav; do
  case "$f" in
    *_a.wav|*_b.wav|*_mix.wav) ;;
    *) samples+=("$f") ;;
  esac
done
real=()
while IFS= read -r line; do
  [[ -n "$line" ]] && real+=("$line")
done < <("$PY" -c '
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    raise SystemExit
for line in p.read_text().splitlines():
    if line.strip():
        print(json.loads(line)["path"])
' "$WORK/audio/score.jsonl")
eval_args=(--out "$OUT/eval.json")
if ((${#samples[@]})); then eval_args+=(--samples "${samples[@]}"); fi
if ((${#real[@]})); then eval_args+=(--real "${real[@]}"); fi
"$PY" "$ROOT/runpod/eval_quick.py" "${eval_args[@]}" || echo "eval failed"

if [[ "$MODE" == local ]]; then
  find "$OUT/logs" -name 's*.stdout' -print 2>/dev/null | while read -r log; do
    echo "---- $log ----"
    tail -n 15 "$log" || true
  done
fi
publish || true
resources || true
status "done"
echo "outputs in $OUT"
