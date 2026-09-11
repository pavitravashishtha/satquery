import numpy as np
# Note: In a real environment we would import rasterio.
# import rasterio

def normalize_pair(optical, sar):
    """
    Per-channel standardization, applied identically at train AND inference.
    optical: numpy array
    sar: numpy array
    """
    # Optical: normalize per channel
    opt = (optical - optical.mean(axis=(0,1), keepdims=True)) / \
          (optical.std(axis=(0,1), keepdims=True) + 1e-6)
          
    # SAR: normalize globally (or per channel if multi-band)
    sar = (sar - sar.mean()) / (sar.std() + 1e-6)
    
    return opt.astype(np.float32), sar.astype(np.float32)
