import numpy as np

def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0.0):
    """
    TODO: implement a cosine annealing learning rate scheduler.
    
    Args:
        initial_lr (float): Initial learning rate.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        min_lr (float): Minimum learning rate to reach.
        
    Returns:
        float: Adjusted learning rate for the current epoch.
    """
    lt=min_lr+ (0.5*(initial_lr-min_lr)*(1+np.cos(np.pi * epoch / total_epochs)))
    return lt

if __name__ == "__main__":
    for e in [0, 25, 50, 75, 100]:
        print(e, cosine_annealing(initial_lr=0.1, epoch=e, total_epochs=100, min_lr=0.001))
    