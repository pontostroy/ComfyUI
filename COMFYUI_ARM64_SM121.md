# ComfyUI on ARM64 SM121 (DGX Spark / GB10) — Normal Install Guide

This guide adapts the container-focused SM121 notes to a **normal local install** (`uv` + `pip`) for this repository.

## Why this exists
The upstream SM121 notes primarily describe Docker build-time patches for xformers. In normal installs, we cannot patch wheel sources during image build, so this fork applies equivalent runtime safety in `comfy/model_management.py` for NVIDIA `sm120+`:

- Auto-set `XFORMERS_DISABLE_FLASH_ATTN=1`.
- Force xformers CUTLASS FMHA ops to self-reject on SM121+.
- Keep default dispatch on safer backends (Sage/PyTorch FA2/SDPA depending on path).

## Recommended install (uv)

```bash
uv venv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements-sm121.txt
uv pip install -U --pre comfyui-manager
```

## Recommended launch flags

```bash
uv run python main.py \
  --fast-mmap-load \
  --cuda-uma \
  --fast-unload-models \
  --use-sage-attention \
  --listen 0.0.0.0 \
  --enable-manager
```

`--use-sage-attention` is recommended on Blackwell even with runtime xformers guards enabled.

## Quick verification

Start ComfyUI once and check startup logs:

- Expect a log line indicating `XFORMERS_DISABLE_FLASH_ATTN=1` was set for NVIDIA sm120+ compatibility.
- If xformers is installed, the runtime SM121 patch should apply and prevent CUTLASS FMHA dispatch on SM121+.

## Notes

- This guide is for **local installs** only.
- For full container build workflows, refer to the upstream DGX docs that this was adapted from.

## Requirements files comparison

- `requirements.txt`: upstream/general ComfyUI dependencies (no SM121-specific `xformers` pin).
- `requirements-sm121.txt`: extends `requirements.txt`, pins `xformers==0.0.32`, and keeps an explicit SM121 dependency block (`einops`, `torchsde`, `kornia`, `spandrel`, `soundfile`, `huggingface_hub[cli]`, `transformers`, `tokenizers`, `sentencepiece`, `safetensors`, `aiohttp`, `yarl`, `psutil`, `tqdm`, `Pillow`, `scipy`, `numpy`, etc.) for easier parity with DGX patch notes.

For SM121 you should install torch/torchvision/torchaudio first from the CUDA index, then install `requirements-sm121.txt`.

## Build xformers from source (optional, ARM64 SM121)

If you need a source-built xformers wheel for your exact environment:

```bash
uv pip uninstall -y xformers
git clone --depth 1 --branch v0.0.32 --recurse-submodules https://github.com/facebookresearch/xformers.git /tmp/xformers-sm121
cd /tmp/xformers-sm121
export XFORMERS_DISABLE_FLASH_ATTN=1
export TORCH_CUDA_ARCH_LIST="8.0;12.1"
uv pip install --no-build-isolation -v .
```

Then run ComfyUI with:

```bash
uv run python main.py --fast-mmap-load --cuda-uma --fast-unload-models --use-sage-attention --enable-manager
```

Notes:
- Build can take a long time on ARM64.
- Runtime SM121 guards in this repo are still applied as a fallback.

## Build SageAttention from source (recommended on SM121)

If you use `--use-sage-attention`, build/install SageAttention in the same environment:

```bash
git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sage
cd /tmp/sage
export TORCH_CUDA_ARCH_LIST="8.0;8.9;12.1"
uv pip install --no-build-isolation -v .
```

Quick check:

```bash
uv run python -c "import sageattention; print('sageattention ok')"
```

## Build PyTorch from source (advanced / usually unnecessary)

Most users should use NVIDIA-provided wheels or NGC images. Build PyTorch only if you have a specific patch requirement.

```bash
uv pip uninstall -y torch torchvision torchaudio
git clone --recursive https://github.com/pytorch/pytorch.git /tmp/pytorch-sm121
cd /tmp/pytorch-sm121
git submodule sync
git submodule update --init --recursive
export TORCH_CUDA_ARCH_LIST="12.1"
export CMAKE_CUDA_ARCHITECTURES=121
export MAX_JOBS=8
uv pip install --no-build-isolation -v -e .
```

Optional follow-up builds:

```bash
uv pip install --no-build-isolation -v git+https://github.com/pytorch/vision.git
```

## Build torchaudio from source (special handling)

For ARM64/SM121 custom stacks, `torchaudio` often needs to be built against your exact local PyTorch build.

```bash
uv pip uninstall -y torchaudio
git clone --recursive https://github.com/pytorch/audio.git /tmp/audio-sm121
cd /tmp/audio-sm121
git submodule sync
git submodule update --init --recursive
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.1"
uv pip install --no-build-isolation -v .
```

If you don't need audio nodes/features, you can skip torchaudio entirely.

Validation:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
uv run python -c "import xformers; print(xformers.__version__)"
uv run python -c "import torchaudio; print(torchaudio.__version__)"
```
