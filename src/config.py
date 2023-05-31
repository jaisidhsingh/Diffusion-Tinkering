from types import SimpleNamespace


cfg = SimpleNamespace(**{})
#################################### DIRECTORIES ################################
cfg.results_dir = "../results"
cfg.ckpt_dir = "../checkpoints"
cfg.log_dir_name = "../logs"
cfg.log_with = "tensorboard"

################################### DATASETS ######################################
cfg.datasets = {
    "lsun-dining-room": {
        "data_dir": "../datasets/lsun/data/images"
	},
    "stanford-cars": {
        "root_dir": "../datasets/stanford-cars/data/images",
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz"
    }
}

################################# HYPERPARAMETERS ###################################
cfg.image_size = 128
cfg.batch_size = 32
cfg.pin_memory = True
cfg.shuffle = True
cfg.num_workers = 1

cfg.timesteps = 1000
cfg.scheduler = "ddpm"

cfg.num_epochs = 50
cfg.learning_rate = 1e-4
cfg.lr_warmup_steps = 500
cfg.save_train_results_point = 10

cfg.mixed_precision = "fp16"
cfg.gradient_accumulation_steps = 1

################################### Unet 2D Config #########################################
cfg.unet_2d_kwargs = dict(
    sample_size=cfg.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

