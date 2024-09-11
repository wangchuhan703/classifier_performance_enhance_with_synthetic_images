import torch
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision.datasets import CIFAR100
from nltk.corpus import wordnet

# Stable Diffusion model
model_path = "/ocean/projects/cis220031p/hchen10/chw_research/stable_diffusion_v1_5"
pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# folder
base_output_folder = "Diffusion100_2"


cifar100_classnames = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                       'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                       'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                       'bottles', 'bowls', 'cans', 'cups', 'plates',
                       'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                       'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                       'bed', 'chair', 'couch', 'table', 'wardrobe',
                       'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                       'bear', 'leopard', 'lion', 'tiger', 'wolf',
                       'bridge', 'castle', 'house', 'road', 'skyscraper',
                       'cloud', 'forest', 'mountain', 'plain', 'sea',
                       'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                       'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                       'crab', 'lobster', 'snail', 'spider', 'worm',
                       'baby', 'boy', 'girl', 'man', 'woman',
                       'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                       'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                       'maple', 'oak', 'palm', 'pine', 'willow',
                       'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                       'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
                       ]

# number of images for each class
num_images = 500


for classname in cifar100_classnames:

    output_folder = os.path.join(base_output_folder, classname)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    prompt = f"a photo of {classname}"

        # Generate and save images
    for i in range(num_images):
        image = pipe(prompt, guidance_scale=2, num_inference_steps=50).images[0]
        image = image.resize((32, 32), Image.LANCZOS)
        file_path = os.path.join(output_folder, f"{classname}_{i + 1}.png")
        image.save(file_path)
        print(f"Saved image {i + 1} at {file_path}")

