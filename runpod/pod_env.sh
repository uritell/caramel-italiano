# GPU venv and Mimi packages. pod_run.sh and lean_bootstrap.sh both call install_pod.
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
