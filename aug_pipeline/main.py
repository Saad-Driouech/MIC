from torch.utils.data import Dataset
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import cv2

class DiffusionAugmentationDataset(Dataset):
    def __init__(self, image_paths, diffusion_pipeline, key_words, negative_prompt, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            diffusion_model: Pre-loaded Stable Diffusion pipeline.
            key_words (list): List of key words for text conditioning.
            negative_prompt (str): Negative prompt for refinement.
            transform (callable, optional): Optional transform to apply to images after augmentation.
        """
        self.image_paths = image_paths
        self.diffusion_pipeline = diffusion_pipeline
        self.key_words = key_words
        self.negative_prompt = negative_prompt
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        original_path = self.image_paths[idx]
        image = self.load_image(original_path)

        # Choose a random key word
        key_word = self.key_words[idx % len(self.key_words)]  # Alternate through keywords

        # Create text prompt
        text_prompt = f"Urban landscape photography, 50mm lens, {key_word}"

        # Generate image
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)

        augmented_image = pipe(
            prompt=text_prompt, 
            num_inference_steps=10, 
            negative_prompt=self.negative_prompt,
            generator=generator, 
            image=image
        ).images[0]

        # Apply additional transforms (if any)
        if self.transform:
            augmented_image = self.transform(augmented_image)

        # Save the image
        directory_path = os.path.dirname(original_path)
        file_name = os.path.basename(original_path)
        save_dir = os.path.join(directory_path, "augmented")
        print(f"Saving augmented image to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_name}")
        augmented_image.save(save_path)

        return augmented_image

    def load_image(self, path):
        # Load the image using PIL and convert to RGB
        image = Image.open(path).convert("RGB")
        image = np.array(image)

        # Get the Canny edge image
        image = cv2.Canny(image, 100, 200)
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
    controlnet = ControlNetModel.from_pretrained("MIC/aug_pipeline/controlnet", torch_dtype=dtype)
    
    # Load diffusion pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype
    )

    # Set scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)

    if device.type == "cuda":
        pipe.enable_model_cpu_offload()

    # Dataset instantiation
    dataset = DiffusionAugmentationDataset(
        image_paths=[
            "MIC/aug_pipeline/images/berlin_000003_000019_leftImg8bit.png", 
            "MIC/aug_pipeline/images/berlin_000058_000019_leftImg8bit.png",
            "MIC/aug_pipeline/images/berlin_000446_000019_leftImg8bit.png"
        ],
        diffusion_pipeline=pipe,
        key_words=["snowy", "golden hour"],
        negative_prompt="monochrome, trees in sky",
        transform=None,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for augmented_image in dataloader:
        pass
