def reshape_transform_gcvit(tensor):
    # GCViTBlock output: (B,H,W,C) -> (B,C,H,W)
    result = tensor.permute(0, 3, 1, 2)
    return result

def reshape_transform_vit(tensor):
    # ViTBlock output: (B,N+1,C) -> skip cls -> (B,C,H,W) 
    tensor = tensor[:, 1:]
    B, N, C = tensor.shape
    h = w = int(N ** 0.5)
    return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)