#!/usr/bin/env python3
""" Text to Image Synthesize Script
"""

import os
import argparse
import math
import json
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from PIL import Image
from typing import List, Tuple, Union
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Text to Image Synthesize Script')
    parser.add_argument('--model', type=str, default='C:/2024_DL/stable_diffusion_v1_5', help='model name')
    parser.add_argument('--text_prompt_json', type=str, default='gpt4o_domainNet_prompts/clipart_domainNet_prompts.json', help='text prompt json file')
    parser.add_argument('--save_dir', type=str, default='test/gpt4o_clipart', help='save directory')
    parser.add_argument('--mp_save_count', type=int, default=4, help='multiprocessing save cpu count')
    parser.add_argument('--guidance_scale', type=float, default=2, help='default guidance scale if no guidance scale in json is provided')
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    parser.add_argument('--num_gen_images_per_class', type=int, default=20, help='number of images per class')
    parser.add_argument('--num_images_per_prompt', type=int, default=5, help='batch size for each prompt')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--xformer_acceleration', default=False, action='store_true', help='use xformer to save memory and accelerate, not working for v100')
    
    parser.add_argument('--start_idx', type=int, default=None, help='start index, used for split the generation process into sub processes')
    parser.add_argument('--end_idx', type=int, default=None, help='start index, used for split the generation process into sub processes')
    args = parser.parse_args()
    return args


def create_gen_model(model_name: str, xformer_acceleration: bool=False):
    # create pipeline
    if model_name == 'C:/2024_DL/stable_diffusion_v1_5':
        pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipeline = pipeline.to("cuda")
    # xformer acceleration
        if xformer_acceleration:
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
            pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            # Workaround for not accepting attention shape using VAE for Flash Attention
            pipeline.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    else:
        raise NotImplementedError(f"model {model_name} not implemented")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    return pipeline


def gen_text2img(model: Union[str, StableDiffusionPipeline, AutoPipelineForText2Image],
                 text_prompt_list: List[str],
                 save_dir: str,
                 guidance_scale: float=2,
                 image_width: int=256,    
                 image_height: int=256,
                 num_inference_steps: int=50,
                 num_images_per_prompt: int=3,
                 num_gen_images_per_class: int=10,
                 xformer_acceleration: bool=False,
                 mp_save_count: int=4,
                 *args, **kwargs):
    
    # create pipeline 
    if isinstance(model, str):
        pipeline = create_gen_model(model, xformer_acceleration)
    else:
        pipeline = model
    
    # generate images
    exsiting_files = sorted(os.listdir(save_dir))
    image_count = len(exsiting_files)

    # determine starting index for prompts
    start_prompt_idx = 0
    if exsiting_files:
        last_file = exsiting_files[-1]
        start_prompt_idx = int(last_file.split('_')[0].replace('prompt', '')) + 1

    num_gen_images_per_prompt = int(math.ceil(num_gen_images_per_class / len(text_prompt_list)))

    for prompt_idx in range(start_prompt_idx, len(text_prompt_list)):
        text_prompt = text_prompt_list[prompt_idx]
        # check existing files
        exsiting_files = [f.split("\\")[-1] for f in glob(os.path.join(save_dir, f"prompt{prompt_idx}_*.jpeg"))]
        exsiting_files = sorted(exsiting_files, key=lambda x: int(x.split('_')[1].replace('img', '').split('.')[0]))
        existing_count = len(exsiting_files)

        while existing_count < num_gen_images_per_prompt:
            num_images_needed = num_gen_images_per_prompt - existing_count
            output = pipeline(text_prompt,
                              guidance_scale=guidance_scale,
                              height=image_height,
                              width=image_width,
                              num_inference_steps=num_inference_steps,
                              num_images_per_prompt=min(num_images_per_prompt, num_images_needed))

            gen_images = output.images

            if hasattr(output, 'nsfw_content_detected'):
                gen_images = [img for img, flag in zip(gen_images, output.nsfw_content_detected) if not flag]

            # # select only sfw images
            # if hasattr(output, 'nsfw_content_detected'):
            #     nsfw_flags = output.nsfw_content_detected
            # elif hasattr(output, 'nsfw_detected'):
            #     nsfw_flags = output.nsfw_detected
            # else:
            #     nsfw_flags = None
            #
            # if nsfw_flags is not None:
            #     filtered_gen_images = []
            #     for gen_image, nsfw_flag in zip(gen_images, nsfw_flags):
            #         if nsfw_flag:
            #             continue
            #         filtered_gen_images.append(gen_image)
            #     gen_images = list(filtered_gen_images)

            # update image count

            for img in gen_images:
                save_path = os.path.join(save_dir, f"prompt{prompt_idx}_img{existing_count}.jpeg")
                img.save(save_path)
                existing_count += 1
                image_count += 1

            if existing_count >= num_gen_images_per_prompt:
                break


        print(f"finished prompt {text_prompt}, generated {image_count} images")

    return image_count



def save_image(image: Image, save_path: str):
    image.save(save_path, "JPEG", quality=100, subsampling=0)


def save_images_multiprocess(mp_save_count, gen_images_list, save_names_list):
    import multiprocessing
    pool = multiprocessing.Pool(mp_save_count)
    pool.starmap(save_image, zip(gen_images_list, save_names_list))
    pool.close()
    pool.join()


def check_image(image_path):
    try:
        img = Image.open(image_path)
    except:
        os.remove(image_path)
        print(f"corrupted image {image_path}")
        return False
    return True


def check_images_multiprocess(mp_save_count, image_paths):
    import multiprocessing
    pool = multiprocessing.Pool(mp_save_count)
    results = pool.map(check_image, image_paths)
    pool.close()
    pool.join()
    return results


def main(args):
    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    save_dir = args.save_dir
    guidance_scale = args.guidance_scale
    num_gen_images_per_class = args.num_gen_images_per_class
    num_images_per_prompt = args.num_images_per_prompt
    image_size = args.image_size
    mp_save_count = args.mp_save_count
    xformer_acceleration = args.xformer_acceleration
    os.makedirs(save_dir, exist_ok=True)
    
    # read text prompt list
    with open(args.text_prompt_json, 'r') as f:
        text_prompt_json = json.load(f)
    
    # save the text prompt json in save_dir 
    with open(os.path.join(save_dir, 'text_prompt_domainnet.json'), 'w') as f:
        json.dump(text_prompt_json, f, indent=4)
    
    # get folder names to be generated from this process
    folder_names = sorted(list(text_prompt_json.keys()))
    if args.start_idx is not None:
        folder_names = folder_names[args.start_idx:args.end_idx]
    
    # create model
    model = create_gen_model(args.model, xformer_acceleration)    
    
    # generate images
    for folder_name in folder_names:
        text_prompt_meta = text_prompt_json[folder_name]

        # get guidance scale and prompt list 
        cls_guidance_scale = text_prompt_meta.get('guidance_scale', guidance_scale)
        cls_prompt_list = text_prompt_meta['prompt']
        modified_prompt_list = [f"A clipart of {folder_name}. {prompt}" for prompt in cls_prompt_list]
        cls_save_dir = os.path.join(save_dir, f"clipart_{folder_name}")
        cls_save_dir = cls_save_dir.replace("\\", "/")
        os.makedirs(cls_save_dir, exist_ok=True)
        print(f"generating {cls_save_dir} with guidance scale {cls_guidance_scale}")
        
        # sanity check
        num_files = len(os.listdir(cls_save_dir))
        if num_files >= num_gen_images_per_class:
            print(f"{cls_save_dir} already has {num_files} files, skip")
            continue

        # generate images
        image_count = gen_text2img(model=model, text_prompt_list=modified_prompt_list, save_dir=cls_save_dir,
                                   image_height=image_size, image_width=image_size,
                                   guidance_scale=cls_guidance_scale, num_gen_images_per_class=num_gen_images_per_class, num_images_per_prompt=num_images_per_prompt,
                                   mp_save_count=mp_save_count, xformer_acceleration=xformer_acceleration)


        print(f"finished {folder_name}, generated {image_count} images")


if __name__ == '__main__':
    # parse args
    args = parse_args()
    main(args)
