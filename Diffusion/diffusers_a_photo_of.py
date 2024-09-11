import torch
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision.datasets import CIFAR10
# from nltk.corpus import wordnet

# Stable Diffusion model
model_path = "C:/2024_DL/stable_diffusion_v1_5"
pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# folder
base_output_folder = "StableDiffusionImages_2_a_photo_of"


cifar10_classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# number of images for each class
num_images = 5


for classname in cifar10_classnames:

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