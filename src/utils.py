from config import cfg

import torch
import os
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt 
import argparse
from diffusers import DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler


def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="ddpm-stanford-cars")
    parser.add_argument("--dataset-name", type=str, default="stanford-cars")
    parser.add_argument("--diffusion-method", type=str, choices=["ddpm"], default="ddpm")
    parser.add_argument("--num-epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    args = parser.parse_args()
    return args    

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_scheduler(scheduler_name):
    mapping = {
        "ddpm": DDPMScheduler,
        "lms": LMSDiscreteScheduler,
        "pndm": PNDMScheduler,
        "euler": EulerDiscreteScheduler
    }
    return mapping[scheduler_name]

def get_noisy_sample(args, image, scheduler, timesteps=torch.tensor([50]).long()):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    
    noise = torch.randn(image.shape)
    noisy_image = scheduler.add_noise(image, noise, timesteps)
    noisy_image = Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

    image = Image.fromarray(((image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]) 

    concat = get_concat_h(image, noisy_image)
    save_dir = os.path.join(cfg.results_dir, args.run_name)
    concat.save(os.path.join(save_dir, "forward_process_noise_example.png"))

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def get_evaluation_images(args, pipeline, epoch):
    images = pipeline(batch_size=args.batch_size, generator=torch.manual_seed(0))
    images = make_grid(images.images, rows=4, cols=4)

    save_dir = os.path.join(cfg.results_dir, args.run_name, "evaluation_samples")
    os.makedirs(save_dir, exist_ok=True)
    images.save(os.path.join(save_dir, f"eval_epoch_{epoch+1}.png"))
    print("Model evaluated and results saved!")
    print(" ")

def save_ckpt(args, pipeline, epoch):
    save_dir = os.path.join(cfg.ckpt_dir, args.run_name, f"pipeline_epoch_{epoch+1}")
    os.makedirs(save_dir, exist_ok=True)
    pipeline.save_pretrained(save_dir)
    print("Diffusion pipeline saved!")
    print(" ")