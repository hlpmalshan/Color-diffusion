# Import necessary libraries
from argparse import ArgumentParser                 # For parsing command-line arguments
import wandb                                        # For experiment tracking and visualization
import os                                           # For interacting with the operating system (e.g., paths)
import glob                                         # For file pattern matching
import torch                                        # PyTorch core library
import pytorch_lightning as pl                      # For simplified PyTorch training
from dataset import ColorizationDataset, make_dataloaders   # Custom dataset and dataloader setup
from model import ColorDiffusion                    # Custom diffusion model
from utils import init_weights, get_device, load_default_configs  # Utility functions
from pytorch_lightning.loggers import WandbLogger   # Wandb logger for PyTorch Lightning
from denoising import Unet, Encoder                 # Denoising model components
from pytorch_lightning.callbacks import ModelCheckpoint     # Callback for saving model checkpoints

def sanitize_config(config):
    '''
    Function to convert configuration dictionaries to a JSON-serializable format.
    This is used to safely log configurations to platforms like WandB.
    '''
    sanitized = {}                              # Initialize an empty dict to hold sanitized config                             
    for key, value in config.items():           # Iterate through all key-value pairs
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):  # Include only JSON-serializable types
            sanitized[key] = value              # Directly add if value is serializable
        elif isinstance(value, (list, dict)):   # If value is list or dict, sanitize recursively
            try:
                sanitized[key] = sanitize_config(value) if isinstance(value, dict) else [sanitize_config(v) if isinstance(v, dict) else v for v in value]
            except:
                pass                            # Skip non-serializable types (e.g., functions, modules, classes)
    return sanitized                            # Return the sanitized dictionary

def main():
    '''
    Main training function for the ColorDiffusion model
    Handles argument parsing, config loading, model setup, checkpointing, and training
    '''
    parser = ArgumentParser(description="Train color diffusion model")          # Create argument parser                          
    parser.add_argument("--log", default=False)                                 # Enable WandB logging
    parser.add_argument("--cpu-only", default=False, action="store_true")       # Force CPU usage
    parser.add_argument("--dataset", default="./dataset/nordland/winter", help="Path to dataset (e.g., ./dataset/nordland/winter)") # Dataset path
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint")      # Optional checkpoint path
    parser.add_argument("--device", default=None, help="Device to use (e.g., cuda:0, cuda:1, cpu). Defaults to cuda:0 if available, else cpu")
    args = parser.parse_args()                                                  # Parse command-line arguments
    print(args)                                                                 # Print parsed arguments
    
    # Determine device based on arguments
    if args.cpu_only:
        device = "cpu"
    elif args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load deafult configs for encoder, unet, and colordiff model from configs/default directory
    enc_config, unet_config, colordiff_config = load_default_configs()
    
    # Update colordiff_config with the selected device
    colordiff_config["device"] = device
    
    # Extract dataset name from path to organize checkpoints 
    dataset_name = os.path.join(*args.dataset.split(os.sep)[1:])    # e.g., dataset/nordland/winter -> nordland/winter
    checkpoint_dir = os.path.join("./checkpoints", dataset_name)    # Checkpoint directory    
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it doesn't exist
    
    # Create train and validation dataloaders
    train_dl, val_dl = make_dataloaders(args.dataset, colordiff_config, num_workers=2)
    
    # Initialize or load model
    # encoder = Encoder(**enc_config)     # Encoder module
    # unet = Unet(**unet_config)          # UNet for denoising
    # added
    encoder = init_weights(Encoder(**enc_config), init='kaiming') 
    unet = init_weights(Unet(**unet_config), init='kaiming')
    ###

    default_ckpt_path = os.path.join(checkpoint_dir, f"epoch=0-step=0.ckpt")    # Default checkpoint path
    
    # Load model from checkpoint if specified
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Resuming training from checkpoint: {args.ckpt}")
        model = ColorDiffusion.load_from_checkpoint(
            args.ckpt,
            strict=True,
            encoder=encoder,
            unet=unet,
            **colordiff_config,
        )
    else:
        print("No checkpoint found, initializing new model")
        model = ColorDiffusion(
            encoder=encoder,
            unet=unet,
            **colordiff_config)
    
    # Move model to the specified device
    model.to(device)
    
    # Initialize WandB logger if enabled
    if args.log:
        wandb_logger = WandbLogger(project="Color_diffusion_Nordland")  # Initialize WandB logger
        wandb_logger.watch(unet)                                        # Log gradients of unet
        # wandb_logger.experiment.config.update(colordiff_config)
        # wandb_logger.experiment.config.update(unet_config)
        
        # Sanitize configs before logging to WandB
        wandb_logger.experiment.config.update(sanitize_config(colordiff_config))
        wandb_logger.experiment.config.update(sanitize_config(unet_config))
    
    # Checkpoint callback to save only the latest checkpoint with epoch=*-step=* naming
    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,     # Save directory
        filename="{epoch}-{step}",  # Filename format
        every_n_train_steps=1,      # 1 - Save after every step
        save_top_k=1,               # Keep only the latest checkpoint
        save_last=False,            # Disable separate 'last.ckpt' to avoid extra files
        #monitor="val_loss"         # Monitoring disabled for step-wise saving
    )
    
    # Set up PyTorch Lightning trainer      
    trainer = pl.Trainer(
        max_epochs=colordiff_config["epochs"],          # Total epochs
        logger=wandb_logger if args.log else None,      # Attach logger if available
        accelerator="gpu" if device.startswith("cuda") else "cpu",  # Use GPU or CPU
        num_sanity_val_steps=1,                         # Validate once before training
        devices=[int(device.split(":")[1])] if device.startswith("cuda") else 1,    # Device index or count
        log_every_n_steps=3,                            # Logging interval
        callbacks=[ckpt_callback],                      # Attach checkpoint callback
        profiler="simple" if args.log else None,        # Simple profiler for WandB
        accumulate_grad_batches=colordiff_config["accumulate_grad_batches"],        # Gradient accumulation
    )
    # Start training
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.ckpt)
        
if __name__ == "__main__":
    main()

'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log", default=False)
    parser.add_argument("--cpu-only", default=False)
    parser.add_argument("--dataset", default="./img_align_celeba", help="Path to unzipped dataset (see readme for download info)")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    print(args)

    enc_config, unet_config, colordiff_config = load_default_configs()
    train_dl, val_dl = make_dataloaders(args.dataset, colordiff_config, num_workers=2, limit=35000)
    colordiff_config["sample"] = False
    colordiff_config["should_log"] = args.log

    #TODO remove 
    # args.ckpt = "/home/ec2-user/Color-diffusion/Color_diffusion_v2/23l96nt1/checkpoints/last.ckpt"
    args.ckpt = "./checkpoints/last.ckpt"

    
    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    if args.ckpt is not None:
        print(f"Resuming training from checkpoint: {args.ckpt}")
        model = ColorDiffusion.load_from_checkpoint(
            args.ckpt, 
            strict=True, 
            unet=unet, 
            encoder=encoder, 
            train_dl=train_dl, 
            val_dl=val_dl, 
            **colordiff_config
            )
    else:
        model = ColorDiffusion(unet=unet,
                               encoder=encoder, 
                               train_dl=train_dl,
                               val_dl=val_dl, 
                               **colordiff_config)
    if args.log:
        wandb_logger = WandbLogger(project="Color_diffusion_v2")
        wandb_logger.watch(unet)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    ckpt_callback = ModelCheckpoint(every_n_train_steps=300, save_top_k=2, save_last=True, monitor="val_loss")

    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger if args.log else None, 
                        accelerator=colordiff_config["device"],
                        num_sanity_val_steps=1,
                        devices= "auto",
                        log_every_n_steps=3,
                        callbacks=[ckpt_callback],
                        profiler="simple" if args.log else None,
                        accumulate_grad_batches=colordiff_config["accumulate_grad_batches"],
                        )
    trainer.fit(model, train_dl, val_dl)
'''