from torch.utils.data import Dataset
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    AutoencoderKL, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler, 
    PNDMScheduler, 
    DPMSolverMultistepScheduler
)
import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import time
import glob

class DiffusionAugmentationDataset(Dataset):
    def __init__(self, images_paths, diffusion_pipeline, key_words, negative_prompt, resize, crop):
        """
        Args:
            images_paths (list): List of image file paths.
            diffusion_model: Pre-loaded Stable Diffusion pipeline.
            key_words (list): List of key words for text conditioning.
            negative_prompt (str): Negative prompt for refinement.
            transform (callable, optional): Optional transform to apply to images after augmentation.
        """
        self.images_paths = images_paths
        self.diffusion_pipeline = diffusion_pipeline
        self.key_words = key_words
        self.negative_prompt = negative_prompt
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])

        self.model_name = self.diffusion_pipeline._name_or_path.split("/")[-1]
        self.scheduler_name = diffusion_pipeline.scheduler.__class__.__name__
        self.timestep_spacing = diffusion_pipeline.scheduler.config.timestep_spacing
        self.num_inference_steps = 50
        self.cfg_scale = 7.5
        self.upscale_factor = 1
        self.resize = resize
        self.crop = crop


    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # Load the image
        original_path = self.images_paths[idx]
        image = self.load_image(original_path)

        # Choose a random key word
        key_word = self.key_words[idx % len(self.key_words)]  # Alternate through keywords

        # Create text prompt
        text_prompt = f"Urban landscape photography, 50mm lens, {key_word}"

        # Generate image
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)

        start_time = time.time()
        augmented_image = pipe(
            prompt=text_prompt, 
            num_inference_steps=self.num_inference_steps, 
            guidance_scale=self.cfg_scale,
            negative_prompt=self.negative_prompt,
            generator=generator, 
            image=image,
            height=int(image.height * self.upscale_factor),
            width=int(image.width * self.upscale_factor)
            ).images[0]
        generation_time = time.time() - start_time
        print(f"Image generation time: {generation_time:.5f} seconds")

        # Save the image
        directory_path = os.path.dirname(original_path)
        file_name = os.path.basename(original_path)
        save_dir = os.path.join(
            directory_path,
            self.model_name,
            f"{self.scheduler_name}_{self.timestep_spacing}",
            f"augmented_{self.num_inference_steps}steps_{self.cfg_scale}scale",
            f"resize_{self.resize}_crop_{self.crop}_{self.upscale_factor}upscale"
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_name}")
        augmented_image.save(save_path)

        # Apply transforms (if any)
        if self.transform:
            augmented_image = self.transform(augmented_image)

        return augmented_image

    def load_image(self, path):
        # Load the image using PIL and convert to RGB
        image = Image.open(path).convert("RGB")

        # Preprocess the image
        preprocess = []
    
        # Add resize only if resize_size is provided
        if self.resize:
            aspect_ratio = image.width / image.height
            target_width, target_height = 1024, int(1024 / aspect_ratio)
            preprocess.append(transforms.Resize((target_width, target_height)))
        
        # Add center crop only if crop_size is provided
        if self.crop:
            preprocess.append(transforms.CenterCrop((512, 512)))
            
        # Apply transformations if any were added
        if preprocess:
            transform = transforms.Compose(preprocess)
            image = transform(image)

        directory_path = os.path.join(os.path.dirname(path), "resize_cropped_test_images")
        os.makedirs(directory_path, exist_ok=True)
        file_name = os.path.basename(path)
        save_dir = os.path.join(directory_path, file_name)
        image.save(save_dir)

        image = np.array(image)

        # Get the Canny edge image
        image = cv2.Canny(image, 50, 100)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        return canny_image

if __name__ == "__main__":
    # Device selection: MPS, CUDA, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print(f"Using CUDA backend: {torch.cuda.get_device_name(0)}")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     dtype = torch.float32
    #     print("Using MPS backend on macOS.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Using CPU as fallback.")

    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained("controlnet", torch_dtype=dtype, use_safetensors=True)

    # Load VAE as suggested by Realistic Vision
    vae = AutoencoderKL.from_single_file(
        "sd-vae-ft-mse-original/diffusion_pytorch_model.safetensors",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    
    # Load diffusion pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V2.0", 
        vae=vae,
        controlnet=controlnet, 
        torch_dtype=dtype,
        revision="refs/pr/4", 
        use_safetensors=True
    )

    # Set scheduler
    scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    scheduler.config.timestep_spacing = 'leading'
    pipe.scheduler = scheduler  

    pipe.to(device)

    if device.type == "cuda":
        pipe.enable_model_cpu_offload()

    # Dataset instantiation
    images_folder = 'images/sim2real/'
    images_paths = glob.glob(f'{images_folder}/*.png')

    dataset = DiffusionAugmentationDataset(
        images_paths=images_paths,
        diffusion_pipeline=pipe,
        key_words=["daytime"],
        negative_prompt="monochrome, trees in sky",
        resize=False,
        crop=False
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for augmented_image in dataloader:
        pass
