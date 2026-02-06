# Model Code

The model code contains the implementation of the Denoising Diffusion Probabilistic Model (DDPM) as well as the U-Net architecture used for the noise prediction. It also contains the implementation of the noise schedulers used for training and inference.

The scripts available are:
- [`diffusion.py`](diffusion.py): Contains the implementation of the DDPM model class, which includes the forward and backward diffusion processes, as well as the training, sampling and inpainting methods.
- [`song`](song) folder: Contains the U-Net class and helper classes and functions from the implementation by Song et al. (see the [Master's thesis](docs/master.hosak.pdf) for more details) 
- [`warmup_cosine.py`](warmup_cosine.py): Contains the implementation of the warmup cosine noise scheduler used for training and inference, as no adequate library was found containing this