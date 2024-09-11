import torch
import os
from PIL import Image
from diffusers import StableDiffusionPipeline

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained("C:/2024_DL/stable_diffusion_v1_5", revision="fp16",
                                               torch_dtype=torch.float16)
# pipe = StableDiffusionPipeline.from_pretrained("/ocean/projects/cis220031p/hchen10/chw_research/stable_diffusion_v1_5",
                                               # revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# 设定生成图片的数量
num_images = 1

# 设定存储图片的文件夹路径
output_folder = "StableDiffusionImages"

# 如果文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 循环生成图片
for i in range(num_images):
    prompt = "A clipart of aircraft carrier. An aircraft carrier is a large naval ship equipped with a full-length flight deck for the launch and recovery of aircraft. It serves as a seagoing airbase, providing a mobile platform for military air operations."
    image = pipe(prompt, guidance_scale=2, height=512,
                              width=512, num_inference_steps=50).images[0]

    # Resize the image to CIFAR-10 dimensions (32x32 pixels)
    # image = image.resize((256, 256), Image.LANCZOS)

    # 构建每张图片的存储路径
    file_path = os.path.join(output_folder, f"image_7.png")

    # 保存图片
    image.save(file_path)
    print(f"Saved image {i + 1} at {file_path}")
