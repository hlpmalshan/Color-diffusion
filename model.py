import torch                                        # Import PyTorch for tensor operations              
from tqdm import tqdm                               # tqdm for progress bars during iterations
from diffusion import GaussianDiffusion             # Gaussian Diffusion process used for noising/denoising steps
from utils import lab_to_pil, lab_to_rgb, show_lab_image, split_lab_channels    # Utility functions for color conversions and visualizations
import torchvision                                  # Torchvision for grid image visualization
import wandb                                        # WandB for experiment tracking and logging
import pytorch_lightning as pl                      # PyTorch Lightning simplifies training loops
from matplotlib import pyplot as plt                # Matplotlib for optional visualizations
from torch_ema import ExponentialMovingAverage      # Exponential Moving Average for stabilizing model weights
from pytorch_lightning.loggers import WandbLogger   # WandB logger for PyTorch Lightning

class ColorDiffusion(pl.LightningModule):
    '''
    LightningModule implementing a conditional diffusion model for colorization tasks.
    Contains methods for forward pass, training loop, validation, testing, sampling,
    Exponential Moving Average (EMA) tracking, and visualization of outputs.
    '''
    def __init__(self,
                 unet,
                 train_dl=None,
                 val_dl=None,
                 encoder=None,
                 loss_fn="l2",
                 T=300,
                 lr=1e-4,
                 batch_size=12,
                 sample=True,
                 should_log=True,
                 using_cond=False,
                 display_every=None,
                 dynamic_threshold=False,
                 use_ema=True,
                 device="cuda:0",
                 **kwargs):
        '''
        Initialize ColorDiffusion model with UNet backbone, optional encoder, EMA, dataloaders,
        optimizer setup, and sampling/visualization preferences.
        '''
        super().__init__()              # Initialize LightningModule base class
        self.unet = unet.to(device)     # Load UNet model to device
        self.T = T                      # Total number of diffusion steps
        self.lr = lr                    # Learning rate
        self.using_cond = using_cond    # Flag for conditioning (not heavily used here)
        self.sample = sample            # Enable sampling during training/validation
        self.should_log = should_log    # Enable logging to WandB
        self.encoder = encoder.to(device) if encoder is not None else None  # Optional encoder, moved to device
        self.display_every = display_every  # Step interval for visualizing samples
        self.val_dl = val_dl            # Validation DataLoader
        self.train_dl = train_dl        # Training DataLoader
        
        # Set loss function based on user input
        if loss_fn == "l1":
            self.loss_fn = torch.nn.functional.l1_loss
        else:
            self.loss_fn = torch.nn.functional.mse_loss

        # Initialize EMA to smooth out training
        self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.9999)
        self.ema.to(device)             # Move EMA to device
        
        # Initialize Gaussian diffusion process with optional dynamic thresholding
        self.diffusion = GaussianDiffusion(T, dynamic_threshold=dynamic_threshold)
        self._device = device           # Internal device tracker
        
        # Set sampling display frequency default
        if sample is True and display_every is None:
            display_every = 1000        # Default visualization frequency
        self.save_hyperparameters(ignore=['unet', 'encoder', 'train_dl', 'val_dl']) # Save configs

    def forward(self, x_noised, t, x_l):
        """
        Performs one denoising step on batch of noised inputs
        Unet is conditioned on timestep and features extracted from greyscale channel
        x_noised: The color image after adding noise (from diffusion process)
        t: The current timestep (how much noise was added)
        x_l: L channel from LAB color space
        """
        cond = self.encoder(x_l.to(self._device)) if self.encoder is not None else None   # Extracts features from the given greyscale image (x_l) if an encoder is given
        noise_pred = self.unet(x_noised.to(self._device), t.to(self._device), greyscale_embs=cond) # UNet predicts the noise based on the given data
        # added
        if torch.isnan(noise_pred).any():
            print("Warning: NaN values in UNet noise prediction")
            noise_pred = torch.nan_to_num(noise_pred, nan=0.0)
        ###
        
        return noise_pred

    def get_batch_pred(self, x_0, x_l):
        """
        Samples a timestep from range [0, T]
        Adds noise to images x_0 to get x_t (x_0 with color channels noised)
        Returns:
        - The model's prediction of the noise, 
        - The real noise applied to the color channels by the forward diffusion process
        """
        # t = torch.randint(0, self.T, (x_0.shape[0],)).to(x_0)
        # x_noised, noise = self.diffusion.forward_diff(x_0, t, T=self.T)
        # return (self(x_noised, t, x_l), noise)
        
        t = torch.randint(0, self.T, (x_0.shape[0],), device=self._device)
        x_noised, noise = self.diffusion.forward_diff(x_0.to(self._device), t, T=self.T)
        return (self(x_noised, t, x_l.to(self._device)), noise)

    def get_losses(self, noise_pred, noise, x_l):
        diff_loss = self.loss_fn(noise_pred, noise)
        return {"total loss": diff_loss}

    def training_step(self, x_0, batch_idx):
        x_l, _ = split_lab_channels(x_0.to(self._device))
        noise_pred, noise = self.get_batch_pred(x_0.to(self._device), x_l)
        losses = self.get_losses(noise_pred, noise, x_l)
        self.log_dict(losses, on_step=True)
        if self.sample and batch_idx and batch_idx % self.display_every == 0 and self.global_step > 1:
            self.test_step(x_0)
        return losses["total loss"]

    def validation_step(self, batch, batch_idx):
        batch = batch.to(self._device)
        x_l, _ = split_lab_channels(batch)
        noise_pred, noise = self.get_batch_pred(batch, x_l)
        losses = self.get_losses(noise_pred, noise, x_l)
        if self.should_log:
            self.log("val_loss", losses["total loss"])
        if self.sample and batch_idx and batch_idx % self.display_every == 0:
            self.sample_plot_image(batch)
        return losses["total loss"]

    @torch.inference_mode()
    def test_step(self, batch, *args, **kwargs):
        batch = batch.to(self._device)
        # x = next(iter(self.val_dl)).to(self._device)
        # self.sample_plot_image(x)
        # self.sample_plot_image(x, use_ema=True)
        self.sample_plot_image(batch)
        self.sample_plot_image(batch, use_ema=True)


    def configure_optimizers(self):
        # learnable_params = list(self.unet.parameters()) \ + list(self.encoder.parameters())
        learnable_params = list(self.unet.parameters())
        if self.encoder is not None:
            learnable_params += list(self.encoder.parameters())
        global_optim = torch.optim.AdamW(learnable_params,
                                         lr=self.lr,
                                         weight_decay=28e-3)
        # return global_optim
        #added
        return {'optimizer': global_optim, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(global_optim), 'monitor': 'val_loss'}, 'gradient_clip_val': 1.0}

    def log_img(self, image, caption="diff samples", use_ema=False):
        image = image.to(self._device)
        rgb_imgs = lab_to_rgb(*split_lab_channels(image))
        # if use_ema:
        #     self.logger.log_image("EMA samples", [rgb_imgs])
        # else:
        #     self.logger.log_image("samples", [rgb_imgs])
        if self.should_log and isinstance(self.logger, WandbLogger):
            if use_ema:
                self.logger.log_image("EMA samples", [rgb_imgs])
            else:
                self.logger.log_image("samples", [rgb_imgs])
        # Silently skip logging if not using WandbLogger or if should_log is False

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()

    @torch.inference_mode()
    def sample_loop(self, x_l, prog=False, use_ema=False, save_all=False):
        """
        Noises color channels to timestep T, then denoises the color channels
        to t=0 to get the colorized image.
        Returns an array containing the noised image,
        intermediate images in the denoising process, and the final image
        """
        x_l = x_l.to(self._device)
        ema = self.ema if use_ema else None
        images = []
        num_images = 13
        img_size = x_l.shape[-1]
        stepsize = self.T // num_images

        # Initialize image with random noise in color channels
        # x_ab = torch.randn((x_l.shape[0], 2, img_size, img_size)).to(x_l)
        x_ab = torch.randn((x_l.shape[0], 2, img_size, img_size), device=self._device)
        img = torch.cat((x_l, x_ab), dim=1)

        counter = range(0, self.T)[::-1]
        if prog:
            counter = tqdm(counter)
        for i in counter:
            # t = torch.full((1,), i, dtype=torch.long).to(img)
            t = torch.full((1,), i, dtype=torch.long, device=self._device)
        
            img = self.diffusion.sample_timestep(self.unet,
                                                 self.encoder,
                                                 img,
                                                 t,
                                                 T=self.T,
                                                 cond=x_l,
                                                 ema=ema)
            # added
            if torch.isnan(img).any():
                print(f"NaN detected at timestep {i}")
                break
            ###

            if i % stepsize == 0:
                # images += img.unsqueeze(0)
                # images.append(img.unsqueeze(0))
                images.append(img)
            if save_all and i % 2 == 0:
                pil_img = lab_to_pil(img)
                pil_img.save(f"./visualization/denoising/{i:04d}.png")
        return images

    @torch.inference_mode()
    def sample_plot_image(self, x_0, show=True, prog=False,
                          use_ema=False, log=True, save_all=False):
        """
        Denoises a single image and displays a grid showing:
        - ground truth image
        - intermediate denoised outputs
        - the final denoised image
        """
        print("Sampling image")
        x_0 = x_0.to(self._device)
        ground_truth_images = []
        if x_0.shape[1] == 3:
            x_l, _ = split_lab_channels(x_0)
            ground_truth_images.append(x_0[:1])
        else:
            x_l = x_0
        x_l = x_l[:1]
        greyscale = torch.cat((x_l, *[torch.zeros_like(x_l)] * 2), dim=1) 
        # ground_truth_images += greyscale.unsqueeze(0)
        # ground_truth_images.append(greyscale.unsqueeze(0))
        ground_truth_images.append(greyscale) # 4D tensor: [batch_size, channels, height, width]
        if len(x_l.shape) == 3:
            x_l = x_l.unsqueeze(0)
        images = ground_truth_images + self.sample_loop(x_l,
                                                        prog=prog,
                                                        use_ema=use_ema,
                                                        save_all=save_all)
        # grid = torchvision.utils.make_grid(torch.cat(images), dim=0).to(x_l)
        grid = torchvision.utils.make_grid(torch.cat(images, dim=0)).to(self._device)
        if show:
            show_lab_image(grid.unsqueeze(0), log=self.should_log)
            print(show)
            # plt.show()
        if self.should_log and log:
            self.log_img(grid.unsqueeze(0), use_ema=use_ema)
        # return lab_to_rgb(*split_lab_channels(images[-1]))
        return images[-1]
