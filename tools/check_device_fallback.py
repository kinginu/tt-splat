import sys, torch, ttnn
# make ANY silent host-fallback RAISE (so we detect ops not running on device)
try:
    ttnn.CONFIG.throw_exception_on_fallback = True
    print("set throw_exception_on_fallback = True", flush=True)
except Exception as e:
    print(f"could not set flag via CONFIG: {e}", flush=True)
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from bin_device import check_buffers, check_inv_device
dev = ttnn.open_device(device_id=0)
try:
    print("-- running bin_to_buffers (dist2+topk+writes) under throw-on-fallback --", flush=True)
    check_buffers(dev, res=128, G=1000, K=128)
    print("-- running build_inv_device (scatter+9 gathers) under throw-on-fallback --", flush=True)
    check_inv_device(dev, res=128, G=1000, K=128)
    print("RESULT: NO FALLBACK -- all binning ttnn ops ran ON DEVICE", flush=True)
except Exception as e:
    print(f"RESULT: FALLBACK DETECTED -> {type(e).__name__}: {str(e)[:240]}", flush=True)
finally:
    ttnn.close_device(dev)
