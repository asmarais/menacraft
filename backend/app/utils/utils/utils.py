def get_device():
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ Using GPU: {device_name}")
    else:
        device = "cpu"
        print("⚠ GPU not available, using CPU")
    return device