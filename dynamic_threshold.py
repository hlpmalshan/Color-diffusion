from utils import right_pad_dims_to
import torch
from einops import rearrange

# def dynamic_threshold(img, percentile=0.8):
#     s = torch.quantile(
#         rearrange(img, 'b ... -> b (...)').abs(),
#         percentile,
#         dim=-1
#     )
#     # If threshold is less than 1, simply clamp values to [-1., 1.]
#     s.clamp_(min=1.)
#     s = right_pad_dims_to(img, s)
#     # Clamp to +/- s and divide by s to bring values back to range [-1., 1.]
#     img = img.clamp(-s, s) / s
#     return img

def dynamic_threshold(img, percentile=0.8, device=None):
    if device is not None:
        img = img.to(device)
    if torch.isnan(img).any():
        print("Warning: NaN values in input to dynamic_threshold")
        img = torch.nan_to_num(img, nan=0.0)
    s = torch.quantile(rearrange(img, 'b ... -> b (...)').abs(), percentile, dim=-1)
    s = torch.clamp(s, min=1.)
    s = right_pad_dims_to(img, s)
    img = img.clamp(-s, s) / s
    return img