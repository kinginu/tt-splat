"""Device selection: CUDA when available, else CPU.

The matrix-native verification code is pure PyTorch, so it runs identically on CPU or GPU. The experiment
entry points (e.g. m05_spike) resolve the device here so runs auto-accelerate on a GPU VM (e.g. an
RTX 3090) — turning the multi-hour CPU sweeps into minutes — with zero algorithm changes. Unit
tests stay on CPU (small, deterministic). Library defaults remain device="cpu"; only the entry
points opt into CUDA, and they pass the device explicitly to BOTH the model and the data/cameras
(model + cameras + images must share a device for render()).
"""
import torch


def default_device(override=None):
    """Resolve the run device. `override` (e.g. 'cuda', 'cpu', 'cuda:0') wins; else CUDA-if-available.
    Raises if CUDA is explicitly requested but unavailable — `torch.device('cuda')` itself never raises
    on construction (only at first alloc), so an explicit check is needed to fail fast."""
    if override:
        dev = torch.device(override)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device '{override}' requested but CUDA is unavailable "
                               "(check GPU passthrough / nvidia-container-toolkit)")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
