from config import cfg
from utils import *
from data import *

import torch
import os
import sys
from diffusers import UNet2DModel, DDPMPipeline 
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")


def train_ddpm(args, cfg):
	# setup directory to save results to
	my_run_dir = os.path.join(cfg.results_dir, args.run_name)
	os.makedirs(my_run_dir, exist_ok=True)

	my_ckpt_dir = os.path.join(cfg.ckpt_dir, args.run_name)
	os.makedirs(my_ckpt_dir, exist_ok=True)

	# load in the dataset
	dataloader, dataset = get_loader(args.dataset_name, "combined", with_dataset=True)
	noise_scheduler = get_scheduler(cfg.scheduler)(num_train_timesteps=cfg.timesteps)

	# load in the UNet model
	model = UNet2DModel(**cfg.unet_2d_kwargs)

	# just a test example of the noisy version of a sample image as part of the forward process
	sample_image = dataset[0]
	get_noisy_sample(args, sample_image, noise_scheduler)

	# setup the training objects
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
	lr_scheduler = get_cosine_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=cfg.lr_warmup_steps,
		num_training_steps=(len(dataloader) * args.num_epochs)
	)

	# setup tensorboard logging
	log_dir = os.path.join(cfg.log_dir_name, args.run_name)
	os.makedirs(log_dir, exist_ok=True)

	# setup training acceleration
	accelerator = Accelerator(
		mixed_precision=cfg.mixed_precision,
		gradient_accumulation_steps=cfg.gradient_accumulation_steps,
		log_with=cfg.log_with,
		project_dir=log_dir
	)
	accelerator.init_trackers(args.run_name)

	dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
		dataloader, model, optimizer, lr_scheduler
	)

	# training loop
	global_steps = 0
	for epoch in range(args.num_epochs):
		bar = tqdm(total=len(dataloader))
		bar.set_description(f"Epoch: {epoch+1}")

		for idx, batch in enumerate(dataloader):
			clean_images = batch

			# sample the noise for the forward process
			noise = torch.randn(clean_images.shape).to(clean_images.device)
			batch_size = clean_images.shape[0]

			# sample a random timestep for each image
			timesteps = torch.randint(
				0, noise_scheduler.config.num_train_timesteps, (batch_size, ), device=clean_images.device
			).long()

			# add noise to the clean images --> forward process
			noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

			with accelerator.accumulate(model):
				# predict the added noise
				noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
				loss = criterion(noise_pred, noise)
				accelerator.backward(loss)

				# clip gradient norms to 1.0 for stability
				accelerator.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
			
			bar.update(1)
			logs = {
				"loss": loss.detach().item(),
				"learning_rate": lr_scheduler.get_last_lr()[0],
				"step": global_steps,
			}
			bar.set_postfix(**logs)
			accelerator.log(logs, step=global_steps)
			global_steps += 1

		if (epoch + 1) % cfg.save_train_results_point == 0:
			pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
			get_evaluation_images(args, pipeline, epoch)
			save_ckpt(args, pipeline, epoch)

	print("Training finished!")	


def train_ldm(args, cfg):
	pass


if __name__ == "__main__":
	args = get_training_args()

	if args.diffusion_method == "ddpm":
		train_ddpm(args, cfg)
	
	if args.diffusion_method == "ldm":
		train_ldm(args, cfg)