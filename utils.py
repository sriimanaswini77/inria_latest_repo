import numpy as np
import cv2

def compute_score(score_map, w, h):
    """
    Compute geometry-averaged score from QATM output.
    
    Args:
        score_map: 2D numpy array of confidence scores
        w: template width
        h: template height
    
    Returns:
        Processed score map with geometry averaging
    """
    # Get original shape
    resized_score_map = score_map
    r_b, r_a = resized_score_map.shape
    
    # Create geometry average kernel
    N = 20
    kernel = np.zeros((2*N+1, 2*N+1))
    for i in range(2*N+1):
        for j in range(2*N+1):
            kernel[i, j] = 1.0 / np.sqrt((i-N)**2 + (j-N)**2 + 1)
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply convolution for geometry averaging
    from scipy.ndimage import convolve
    avg_map = convolve(resized_score_map, kernel, mode='constant', cval=0.0)
    
    return avg_map

def normalize_score(score_map):
    """Normalize score map to [0, 1] range"""
    min_val = score_map.min()
    max_val = score_map.max()
    if max_val - min_val > 0:
        return (score_map - min_val) / (max_val - min_val)
    return score_map
