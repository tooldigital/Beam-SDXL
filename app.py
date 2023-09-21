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

def set_style(prompt, style):

    final_prompt =f""
    negative_prompt = f""

    if style == "3D Model":
        final_prompt= f"professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting"
        negative_prompt = f"ugly, deformed, noisy, low poly, blurry, painting"
    elif style == "Analog Film":
        final_prompt= f"analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage"
        negative_prompt = f"painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    elif style == "Anime":
        final_prompt= f"anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed"
        negative_prompt = f"photo, deformed, black and white, realism, disfigured, low contrast"
    elif style == "Cinematic":
        final_prompt= f"cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
        negative_prompt = f"anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    elif style == "Pencil Sketch":
        final_prompt= f"black and white, pencil sketch {prompt}, graphic, rough, lines"
        negative_prompt = f"photo, deformed, realism, disfigured, deformed, glitch, noisy, realistic"
    elif style == "Comic Book":
        final_prompt= f"comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed"
        negative_prompt = f"photograph, deformed, glitch, noisy, realistic, stock photo"
    elif style == "Digital Art":
        final_prompt= f"concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed"
        negative_prompt = f"photo, photorealistic, realism, ugly"
    elif style == "Line Art":
        final_prompt= f"cline art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics"
        negative_prompt = f"anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
    elif style == "Lowpoly":
        final_prompt= f"low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition"
        negative_prompt = f"noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
    elif style == "Photographic":
        final_prompt= f"cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"
        negative_prompt = f"drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
    elif style == "Abstract":
        final_prompt= f"Abstract style {prompt} . Non-representational, colors and shapes, expression of feelings, imaginative, highly detailed"
        negative_prompt = f"realistic, photographic, figurative, concrete"
    elif style == "Watercolor":
        final_prompt= f"Watercolor painting {prompt} . Vibrant, beautiful, painterly, detailed, textural, artistic"
        negative_prompt = f"anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
    elif style == "Monochrome":
        final_prompt= f"Monochrome {prompt} . Black and white, contrast, tone, texture, detailed"
        negative_prompt = f"colorful, vibrant, noisy, blurry, deformed"
    else:
        final_prompt =prompt

    return final_prompt,negative_prompt

def load_models():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
   
    refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", use_safetensors=True, torch_dtype=torch.float16, variant="fp16")
    refiner.enable_xformers_memory_efficient_attention()
    refiner = refiner.to(device)

    return pipe, refiner

@app.rest_api(loader=load_models, keep_warm_seconds=300)
def doSDXL(**inputs):
    # Grab inputs passed to the API

    pipe, refiner = inputs["context"]

    prompt = inputs["prompt"]
    style = inputs["style"]
    seed = inputs["seed"]
    w = inputs["width"]
    h = inputs["height"]


    if(seed == 0):
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator().manual_seed(seed)

    fprompt, fnegprompt = set_style(prompt,style)


    int_image = pipe(fprompt, prompt_2="", negative_prompt=fnegprompt, negative_prompt_2="", num_inference_steps=50, height=h, width=w, guidance_scale=10, num_images_per_prompt=1, generator=generator, output_type="latent").images
 
    image = refiner(prompt=prompt, prompt_2="", negative_prompt=fnegprompt, negative_prompt_2="", image=int_image).images[0]   
    
    buffered = BytesIO()
    image.save(buffered, format='JPEG',quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    print(fprompt)
    print(fnegprompt)

    return {"image": img_str}


if __name__ == "__main__":
    print("main called")
    # You can customize this query however you want:
    # urls = ["https://www.nutribullet.com"]
    # query = "What are some use cases I can use this product for?"
    # start_conversation(urls=urls, query=query)
