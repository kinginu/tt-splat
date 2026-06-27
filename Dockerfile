# Pin a concrete release tag — NEVER :latest-rc, for reproducibility.
# On the HARDWARE machine, match tt-metal to your host KMD/firmware (`tt-smi`).
ARG TT_METAL_TAG=v0.70.0
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:${TT_METAL_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs build-essential cmake ninja-build ccache \
      gdb clang-format clang-tidy clangd \
      python3-dev python3-pip python3-venv \
      curl wget jq vim less ripgrep sudo ca-certificates tmux \
    && rm -rf /var/lib/apt/lists/*

# NOTE: this image is the execution environment only.

# --- ttsim: virtual Blackhole for kernel correctness WITHOUT silicon ---
# Repo: github.com/tenstorrent/ttsim. Default is the validated release;
# compose.yaml passes the same value. Escape hatch: build with TTSIM_VERSION=vX.Y to SKIP the ttsim
# download (e.g. a pure-CPU image that doesn't need the simulator).
# x86_64 -> libttsim_bh.so   |   aarch64 -> libttsim_bh_aarch64.so
ARG TTSIM_VERSION=v1.8.0
# The ttsim release ships ONLY the .so; tt-metal looks for soc_descriptor.yaml *beside* it
# (path derived from TT_METAL_SIMULATOR). We stage the Blackhole descriptor from the tt-metal
# wheel (verified 2026-06-17 in the built image). The sim also lacks fast-dispatch command-queue
# host channels, so the `sim` service runs TT_METAL_SLOW_DISPATCH_MODE=1 (set in compose.yaml).
RUN if [ "$TTSIM_VERSION" = "vX.Y" ]; then \
      echo ">> TTSIM_VERSION placeholder — skipping ttsim download (set a real version first)"; \
    else \
      mkdir -p /opt/ttsim && \
      wget -O /opt/ttsim/libttsim_bh.so \
        "https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_bh.so" && \
      cp /opt/venv/lib/python3*/site-packages/ttnn/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
         /opt/ttsim/soc_descriptor.yaml; \
    fi
ENV TTSIM_BH_LIB=/opt/ttsim/libttsim_bh.so

# The release image's venv (/opt/venv) ships numpy + pillow but NOT torch
# (verified 2026-06-17 inside the built image: no torch/torchvision/scipy/skimage;
# the tt-metalium *release* image is the C++ runtime, not the PyTorch ML stack).
# The CPU reference is pure-CPU PyTorch (autograd reference + SSIM), so add a pinned
# CPU-only torch + SSIM layer here. The venv doesn't expose pip as a module by
# default -> bootstrap ensurepip first, then install.
RUN python3 -m ensurepip --upgrade 2>/dev/null || true \
 && python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
 && python3 -m pip install --no-cache-dir pytorch-msssim==1.0.0

# Test harness (separate layer so the heavy torch layer stays cached on edits).
RUN python3 -m pip install --no-cache-dir pytest==8.3.3

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
CMD ["/bin/bash"]