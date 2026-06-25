import os

# Must be set before torch is imported anywhere: lets the MPS backend drop
# unsupported audio ops (STFT/mel frontends) to CPU instead of raising. No-op
# on CUDA/CPU. The Spark uses CUDA, where every op is native.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
