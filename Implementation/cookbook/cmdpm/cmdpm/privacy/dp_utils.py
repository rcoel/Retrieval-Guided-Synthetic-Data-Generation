import torch
from torch.distributions.laplace import Laplace

def add_laplace_noise(data, epsilon, sensitivity):
    """
    Add Laplace noise to data for differential privacy.
    
    Args:
        data (torch.Tensor): Input data.
        epsilon (float): Privacy parameter.
        sensitivity (float): Sensitivity of the function.
    
    Returns:
        torch.Tensor: Noisy data.
    """
    scale = sensitivity / epsilon
    noise = Laplace(0, scale).sample(data.shape)
    return data + noise.squeeze()

def calculate_sensitivity(data, norm="L1"):
    """
    Calculate sensitivity of the data.
    
    Args:
        data (torch.Tensor): Input data.
        norm (str): Norm to use for sensitivity calculation ("L1" or "L2").
    
    Returns:
        float: Sensitivity value.
    """
    if norm == "L1":
        return torch.norm(data, p=1).item()
    elif norm == "L2":
        return torch.norm(data, p=2).item()
    else:
        raise ValueError(f"Unsupported norm: {norm}")