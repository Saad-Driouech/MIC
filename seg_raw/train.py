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
        self.image_resize = T.Resize((512, 1024), interpolation=Image.BILINEAR)
        self.mask_resize = T.Resize((512, 1024), interpolation=Image.NEAREST)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace("_augmented", "").replace(".png", "_labelTrainIds.png"))

        image = Image.open(img_path).convert("RGB")
        with Image.open(mask_path) as img:
            mask = np.array(img).astype(np.uint8)

        image = self.image_resize(image)

        mask = Image.fromarray(mask)  # convert to PIL first
        mask = self.mask_resize(mask)
        mask_np = np.array(mask).astype(np.uint8)

        encoding = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()

        label = torch.from_numpy(mask_np).long()

        # unique_vals, counts = np.unique(mask_np, return_counts=True)
        # print("Mask value counts:")
        # for val, count in zip(unique_vals, counts):
        #     print(f"Value {val}: {count} pixels")
        # mapped_mask = map_labels(mask_np)
        # label = torch.from_numpy(mask_np).long()

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }

# --------------------
# Training
# --------------------
def train(image_dir, mask_dir, num_epochs=20, batch_size=4, lr=5e-5, save_path="segformer_gta19.pt"):
    config = SegformerConfig(
        num_labels=19,
        ignore_index=255,
        hidden_size=256,
        num_attention_heads=[4] * 12,
        num_hidden_layers=12,
    )

    model = SegformerForSemanticSegmentation(config)
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

    dataset = GTADataset(image_dir, mask_dir, feature_extractor)
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

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# --------------------
# Run from CLI
# --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to GTA5 images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to GTA5 masks")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_path", type=str, default="segformer_gta19.pt")
    args = parser.parse_args()

    train(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
