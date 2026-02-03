def window_partition(x, window_size):
    """
        inp: (B, H, W, C)
        out: (num_windows * B, window_size, window_size, C)
    """
    B,H,W,C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    
    windows = x.permute(0,1,3,2,4,5).reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H , W):
    """
        int: (num_windows * B, window_size**2, C)
        out: (B, H, W, C)
    """
    
    C = windows.shape[-1]
    x = windows.view(-1, H//window_size, W//window_size, window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).reshape(-1,H,W,C)
    
    return x