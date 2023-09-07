from beam import App, Runtime, Image, Output, Volume

import os
import random

import numpy as np
import PIL.Image
import torch
from diffusers import DiffusionPipeline

import base64
from io import BytesIO

cache_path = "./models"
MAX_SEED = np.iinfo(np.int32).max

app = App(
    name="sdxl",
    runtime=Runtime(
        cpu=8,
        memory="32Gi",
        gpu="A10G",
        image=Image(
            python_version="python3.10",
            python_packages=[
                "accelerate==0.21.0",
                "diffusers==0.19.3",
                "invisible-watermark==0.2.0",
                "Pillow==10.0.0",
                "torch==2.0.1",
                "transformers==4.31.0",
                "xformers==0.0.21",
                "opencv-python"
            ],
            commands=["apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"],
        ),
    ),
    volumes=[Volume(name="models", path="./models")],
)

def load_models():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
   
    refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16")
    refiner.enable_xformers_memory_efficient_attention()
    refiner = refiner.to(device)

    return pipe, refiner

@app.rest_api(loader=load_models)
def doSDXL(**inputs):
    # Grab inputs passed to the API

    pipe, refiner = inputs["context"]

    prompt = inputs["prompt"]

    seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    
    int_image = pipe(prompt, prompt_2="", negative_prompt="", negative_prompt_2="", num_inference_steps=50, height=1024, width=1024, guidance_scale=10, num_images_per_prompt=1, generator=generator, output_type="latent").images
 
    image = refiner(prompt=prompt, prompt_2="", negative_prompt="", negative_prompt_2="", image=int_image).images[0]   
    
    buffered = BytesIO()
    image.save(buffered, format='JPEG',quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"pred": img_str}


if __name__ == "__main__":
    print("main called")
    # You can customize this query however you want:
    # urls = ["https://www.nutribullet.com"]
    # query = "What are some use cases I can use this product for?"
    # start_conversation(urls=urls, query=query)
