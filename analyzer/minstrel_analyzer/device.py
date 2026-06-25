import torch


def select_device() -> torch.device:
    """CUDA on the Spark, MPS on the Mac, CPU as the floor.

    Code stays identical across all three; only the chosen device differs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
