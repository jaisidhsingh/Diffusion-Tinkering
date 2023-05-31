from utils import *
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from tqdm.auto import tqdm
import argparse
from types import SimpleNamespace
import warnings
warnings.simplefilter("ignore")


def inference_ddpm():
    pass

def inference_stable_diffusion(prompt, cfg, extension):
	# load in the models and objects
	# image -> latent space
	vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(cfg.device)

	# get text embedding to condition the UNet
	tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
	text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(cfg.device)

	# UNet for conditioned diffusion over image latents
	unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(cfg.device)

	scheduler = get_scheduler("lms").from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

	text_input = tokenizer(
		prompt, 
		padding="max_length",
		max_length=tokenizer.model_max_length,
		truncation=True,
		return_tensors="pt"
	)

	# make the CLIP text embeddings for conditioning the latent
	with torch.no_grad():
		text_embeddings = text_encoder(text_input.input_ids.to(cfg.device))[0]
		
	max_length = text_input.input_ids.shape[-1]
	uncond_inputs = tokenizer(
		[""] * cfg.batch_size, 
		padding="max_length", 
		max_length=max_length, 
		return_tensors="pt"
	) 

	with torch.no_grad():
		uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(cfg.device))[0]   

	# use both conditioned and unconditioned embeddings, as we have classifier free-guidance
	text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

	# first make random noisy latents, and then denoise them, conditioned on the semantic information in the text embedding
	latents = torch.randn(
		(cfg.batch_size, unet.in_channels, cfg.height // 8, cfg.width // 8),
		generator=cfg.generator,
	)
	latents = latents.to(cfg.device)

	# set the timesteps for the backward process
	scheduler.set_timesteps(cfg.num_inference_steps)
	latents = latents * scheduler.init_noise_sigma

	# the backward process loop to get the image
	for t in tqdm(scheduler.timesteps):
		# expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
		latent_model_input = torch.cat([latents] * 2)

		latent_model_input = scheduler.scale_model_input(latent_model_input, t)

		# predict the noise residual
		with torch.no_grad():
			noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

		# perform guidance
		noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
		noise_pred = noise_pred_uncond + cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

		# compute the previous noisy sample x_t -> x_t-1
		latents = scheduler.step(noise_pred, t, latents).prev_sample

	# scale and decode the image latents with vae
	latents = 1 / 0.18215 * latents

	with torch.no_grad():
		image = vae.decode(latents).sample

	# images = ((image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).cpu().numpy()[0]
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
	images = (image * 255).round().astype("uint8")
	pil_images = [Image.fromarray(image) for image in images]
	
	save_dir = os.path.join(cfg.results_dir, cfg.run_name)
	os.makedirs(save_dir, exist_ok=True)

	pil_images[0].save(os.path.join(save_dir, f"{cfg.run_name}_{extension}.png"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--diffusion-method",
		type=str,
		default="stable_diffusion",
		choices=["ddpm", "stable_diffusion"]
	)
	parser.add_argument(
		"--sd-prompt",
		type=str,
		default="a half open laptop",
		help="Only used for stable diffusion inference."
	)
	args = parser.parse_args()

	if args.diffusion_method == "ddpm":
		inference_ddpm()
	if args.diffusion_method == "stable_diffusion":
		# setup inference configs
		# WORKS ONLY FOR BATCH SIZE 1 RIGHT NOW
		cfg = SimpleNamespace(**{})

		cfg.run_name = "inpainting-content"                 # the name of the experiment run
		cfg.results_dir = "../inference"        # the directory to save results in
		cfg.height = 512                        # default height of Stable Diffusion
		cfg.width = 512                         # default width of Stable Diffusion
		cfg.num_inference_steps = 150           # Number of denoising steps (scheduler dependant)
		cfg.guidance_scale = 7.5                # Scale for classifier-free guidance
		cfg.generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
		cfg.batch_size = 1                      # The batch size controls the number of images you get
		cfg.device = "cuda"                     # Whether or not to use GPU (recommended)

		prompts = [
			"a laptop",
			"a bouquet of tulips",
			"a curled up cat",
			"a pile of clothes",
			"an open book"
		]
		extensions = [
			"laptop",
			"bouquet",
			"cat",
			"clothes",
			"book"
		]
		for i in range(5):
			# get the image
			inference_stable_diffusion(prompts[i], cfg, extension=extensions[i])