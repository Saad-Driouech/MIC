import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerFeatureExtractor
from torch.optim import AdamW
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
import os
import numpy as np

# --------------------
# Dataset
# --------------------
class GTADataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.feature_extractor = feature_extractor
        self.resize = T.Resize((512, 1024), interpolation=Image.BILINEAR)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace("_augmented", "").replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.resize(image)
        mask = self.resize(mask)

        encoding = self.feature_extractor(images=image, return_tensors="pt", padding=True)
        pixel_values = encoding["pixel_values"].squeeze()
        label = torch.from_numpy(np.array(mask)).long()

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }

# --------------------
# Training
# --------------------
def train(image_dir, label_dir, num_epochs=20, batch_size=4, lr=5e-5, save_path="segformer_gta.pt"):
    # Create a SegFormer configuration (this doesn't load any pretrained weights)
    config = SegformerConfig(
        num_labels=35,  # Adjust this based on your dataset
        hidden_size=256,  # SegFormer has different size variants
        num_attention_heads=[4] * 12,  # List for the number of attention heads per layer (example: 4 heads for each layer)
        num_hidden_layers=12,  # Adjust layers as per the variant you're using (base, large, etc.)
    )
    
    # Initialize the model with the configuration (no pretrained weights)
    model = SegformerForSemanticSegmentation(config)

    # Feature extractor (can use default or create your own based on GTA data)
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

    dataset = GTADataset(image_dir, label_dir, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0

        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to segmentation masks")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_path", type=str, default="segformer_gta.pt")
    args = parser.parse_args()

    train(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
