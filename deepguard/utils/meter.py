class AverageMeter(object):
    """Keeps track of average and current values for metrics during training"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all meter statistics"""
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val: float, n: int =1):
        """Update the meter with a new value.
        
        Args:
            val (float): The new value to update
            n (int): The number of samples the value represents.
        
        """
        
        self.val = val 
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count