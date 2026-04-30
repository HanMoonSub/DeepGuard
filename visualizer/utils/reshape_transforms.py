def reshape_transform_gcvit(tensor):
    # (B,H,W,C) -> (B,C,H,W)
    result = tensor.permute(0, 3, 1, 2)
    return result