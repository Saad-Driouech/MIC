from torch.utils.data import Dataset
from transformers import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline

class DiffusionAugmentationDataset(Dataset):
    def __init__(self, image_paths, labels, diffusion_model, key_words, negative_prompt, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of labels corresponding to images.
            diffusion_model: Pre-loaded Stable Diffusion pipeline.
            key_words (list): List of key words for text conditioning.
            negative_prompt (str): Negative prompt for refinement.
            transform (callable, optional): Optional transform to apply to images after augmentation.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.diffusion_model = diffusion_model
        self.key_words = key_words
        self.negative_prompt = negative_prompt
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image = self.load_image(self.image_paths[idx])  # Custom function to read image.
        label = self.labels[idx]

        # Choose a random key word
        key_word = self.key_words[idx % len(self.key_words)]  # Alternate through keywords

        # Create text prompt
        text_prompt = f"Urban landscape photography, 50mm lens, {key_word}"

        # Apply diffusion augmentation
        augmented_image = self.diffusion_model(
            prompt=text_prompt,
            negative_prompt=self.negative_prompt,
            image=image,
        ).images[0]  # Get the augmented output

        # Apply additional transforms (if any)
        if self.transform:
            augmented_image = self.transform(augmented_image)

        return augmented_image, label

    def load_image(self, path):
        # Custom function to load image (e.g., using PIL or OpenCV)
        from PIL import Image
        return Image.open(path).convert("RGB")

# Example usage

if __name__ == "__main__":
    # Load diffusion pipeline
    diffusion_model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

    # Dataset instantiation
    dataset = DiffusionAugmentationDataset(
        image_paths=["img1.jpg", "img2.jpg"],
        labels=[0, 1],
        diffusion_model=diffusion_model,
        key_words=["snowy", "golden hour"],
        negative_prompt="monochrome, trees in sky",
        transform=None,
    )
