import glob
import os
import torch
import torchvision
import numpy as np
from dataset import ColorizationDataset, make_dataloaders
from denoising import Unet, Encoder
from utils import get_device, lab_to_rgb, load_default_configs, \
                    split_lab_channels
from model import ColorDiffusion
from argparse import ArgumentParser

def sanitize_config(config):
    """Create a JSON-serializable copy of the config dictionary."""
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            try:
                sanitized[key] = sanitize_config(value) if isinstance(value, dict) else [sanitize_config(v) if isinstance(v, dict) else v for v in value]
            except:
                pass  # Skip non-serializable nested structures
    return sanitized

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image-path", required=True, dest="img_path")
    parser.add_argument("-T", "--diffusion-steps", default=350, dest="T")
    parser.add_argument("--image-size", default=64, dest="img_size", type=int)
    parser.add_argument("--checkpoint", required=True, dest="ckpt", help="Path to checkpoint")
    parser.add_argument("--show", default=True)
    parser.add_argument("--save", default=True)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--device", default=None, help="Device to use (e.g., cuda:0, cuda:1, cpu). Defaults to cuda:0 if available, else cpu")
    args = parser.parse_args()
    assert args.ckpt is not None, "No checkpoint passed and ./checkpoints/ folder empty"

    # Determine device
    if args.device:
        device = args.device
    else:
        device = get_device()  # Fallback to utils.get_device()
    print(f"Using device: {device}")
    
    enc_config, unet_config, colordiff_config = load_default_configs()
    print("loaded default model config")
    
    colordiff_config["T"] = args.T
    colordiff_config["img_size"] = args.img_size
    colordiff_config["device"] = device
    colordiff_config["should_log"] = False  # Disable logging for inference
    colordiff_config = sanitize_config(colordiff_config)  # Sanitize config

    dataset = ColorizationDataset([args.img_path],
                                  split="val",
                                  config=colordiff_config)
    image = dataset[0].unsqueeze(0)


    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)

    #added

    from PIL import Image
    img = Image.open(args.img_path)
    if img.mode not in ["RGB", "L"]:
        raise ValueError(f"Unsupported image mode: {img.mode}. Expected RGB or L.")
    img_array = np.array(img.convert("RGB"))
    if img_array.max() > 255 or img_array.min() < 0:
        print(f"Warning: Image pixel values out of range [0, 255]: min={img_array.min()}, max={img_array.max()}")

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    for key, value in checkpoint['state_dict'].items():
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"Warning: NaN or Inf values found in checkpoint weight {key}")
    ####
    
    model = ColorDiffusion.load_from_checkpoint(args.ckpt,
                                                strict=True,
                                                unet=unet,
                                                encoder=encoder,
                                                train_dl=None,
                                                val_dl=None,
                                                map_location = device,
                                                **colordiff_config)
    model.to(device)

    colorized = model.sample_plot_image(image.to(device),
                                        show=args.show,
                                        prog=True)
    
    # Check for NaNs before denormalization
    # print("Colorized LAB min/max before denormalization:", colorized.min().item(), colorized.max().item())
    # if torch.isnan(colorized).any():
    #     raise ValueError("Colorized tensor contains NaN values after sampling")
    
    # # Denormalize LAB before RGB conversion (fix for black output)
    # colorized = colorized.clone()  # Clone to avoid in-place modification
    # colorized[:, 0, :, :] = colorized[:, 0, :, :] * 100  # Denormalize L (0-1 -> 0-100)
    # colorized[:, 1:, :, :] = colorized[:, 1:, :, :] * 128  # Denormalize AB (-1-1 -> -128-128)
    # # Clamp to valid LAB ranges to prevent invalid values
    # colorized[:, 0, :, :] = torch.clamp(colorized[:, 0, :, :], 0, 100)
    # colorized[:, 1:, :, :] = torch.clamp(colorized[:, 1:, :, :], -128, 128)

    # print("Colorized LAB min/max after denormalization:", colorized.min().item(), colorized.max().item())

    rgb_img = lab_to_rgb(*split_lab_channels(colorized))
    print("RGB min/max:", rgb_img.min(), rgb_img.max())

    if args.save:
        if args.save_path is None:
            save_path = args.img_path[:-4] + "-colorized.png"
        else:
            save_path = args.save_path + os.path.basename(args.img_path)[:-4] + "-colorized.png"
        save_img = torch.tensor(rgb_img[0], dtype=torch.float32).permute(2, 0, 1)
        
        # Ensure rgb_img is a tensor in [0, 1] for save_image
        # if isinstance(rgb_img, np.ndarray):
        #     save_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)
        #     if save_img.max() > 1.0:  # Normalize if in [0, 255]
        #         save_img = save_img / 255.0
        # else:
        #     save_img = rgb_img.permute(2, 0, 1)  # Assume already a tensor in [0, 1]
        '''
        # Handle 4D tensor (batch dimension) and normalize for save_image
        if isinstance(rgb_img, np.ndarray):
            save_img = torch.tensor(rgb_img, dtype=torch.float32)
            if save_img.dim() == 4:
                save_img = save_img[0]  # Select first image from batch
            save_img = save_img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            if save_img.max() > 1.0:  # Normalize if in [0, 255]
                save_img = save_img
        else:
            save_img = rgb_img
            if save_img.dim() == 4:
                save_img = save_img[0]  # Select first image from batch
            save_img = save_img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            if save_img.max() > 1.0:  # Normalize if in [0, 255]
                save_img = save_img
        '''
        torchvision.utils.save_image(save_img, save_path)
        print(f"Saved colorized image to {save_path}")