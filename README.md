# Implementing Generative Diffusion Models and Concepts
This is a repository containing my learning and implementation of popular diffusion pipelines with the 
<a href="https://huggingface.co/docs/diffusers/index">``diffusers``</a> library.
I also dabble in in-painting experiments with <a href="https://arxiv.org/pdf/2301.07093.pdf">GLIGEN</a>.

### Requirements
> PyTorch

> Diffusers

> Accelerate

> Transformers (huggingface)


### Diffusion Todo:
- [x] DDPM training and results (Stanford Cars)

- [x] DDPM inference (Stanford Cars) 

- [x] Stable Diffusion v-1.4 inference

- [ ] LDM training and conditioning on custom embeddings


### Denoising Diffusion Probabilistic Models
#### Instructions
To train a DDPM on a dataset (find prepped datasets in ``config.py``):

``
cd ./src
``

``
python3 train.py --diffusion-method=ddpm --dataset-name=<your dataset> --run-name=<name of the training run>
``

By default, ``--dataset-name`` is set to ``"stanford-cars"``, ``--diffusion-method`` is set to ``ddpm``.

#### Results
Results of training a DDPM on the Stanford Cars dataset, using a Quadro RTX 5000 (16 GB VRAM). A sample of a noised sample at timestep 50 
using the linear scheduler is shown below:

<img src="results/ddpm-stanford-cars/forward_process_noise_example.png">

Epoch 10:

<img src="results/ddpm-stanford-cars/evaluation_samples/eval_epoch_10.png">

Epoch 30:

<img src="results/ddpm-stanford-cars/evaluation_samples/eval_epoch_30.png">

Epoch 50:

<img src="results/ddpm-stanford-cars/evaluation_samples/eval_epoch_50.png">

### Stable Diffusion v-1.4
#### Instructions
To generate an image from a given text prompt using SD, first ``cd src`` and then:

``
python3 inference.py --diffusion-method=stable-diffision --sd-prompt="your prompt"
``

#### Results
By setting ``--sd-prompt`` as ``A realistic LSUN-like bedroom scene`` we get:

<img src="inference/lsun-bedroom-scenes/lsun-bedroom-scenes_1_4.png">

### Image Inpainting
Used ``gligen_experiments/inpainting_inference.py`` to add the following in LSUN bedroom scenes:

#### Objects
> a laptop
<img src="inference/inpainting-content/inpainting-content_laptop.png">

> a cat
<img src="inference/inpainting-content/inpainting-content_cat.png">

> a book
<img src="inference/inpainting-content/inpainting-content_book.png">

> a bouquet of flowers
<img src="inference/inpainting-content/inpainting-content_bouquet.png">

> a pile of clothes
<img src="inference/inpainting-content/inpainting-content_clothes.png">

#### Results
An example of inpainting the above objects at the location:

<img src="annotations/masks/lsun-bedroom-scenes_1_2.png">

yields these results:

<img src="gligen_inpainting/bedroom_2_results.png">
