def resolve_device(device=None):
    if device and str(device).lower() != "auto":
        return str(device)
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"
