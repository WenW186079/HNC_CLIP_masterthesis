
import os
import json
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# logger 
logger = logging.getLogger("GQA_CLIP_Logger")
logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Paths
json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_clean_strict_val.json'
image_folder_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images'

# Load JSON file
print('start ---')
logger.info("Loading annotations JSON file...")
with open(json_file_path, 'r') as f:
    annotations = json.load(f)
logger.info(f"Loaded {len(annotations)} annotations.")

# Custom Dataset for CLIP
class GQACLIPDataset(Dataset):
    def __init__(self, annotations, image_folder, transform=None):
        self.annotations = annotations
        self.image_folder = image_folder
        self.transform = transform
        self.data_pairs = []

        # Create (image_path, caption) pairs
        logger.info("Creating image-caption pairs...")
        missing_images_count = 0
        
        # Iterate over the annotations (image IDs as keys)
        for img_id, data in annotations.items():
            # Convert image ID to image filename (e.g., "2386621" -> "2386621.jpg")
            image_filename = f"{img_id}.jpg"
            image_path = os.path.join(self.image_folder, image_filename)

            if os.path.exists(image_path):
                # Extract "captions" dictionary
                captions_dict = data.get("captions", {})
                if not captions_dict:
                    logger.warning(f"No captions found for image ID {img_id}. Skipping...")
                    continue

                # Iterate over caption entries (e.g., "48", "49", ...)
                for caption_id, caption_data in captions_dict.items():
                    caption = caption_data.get('caption')  # Get caption text
                    if caption:
                        self.data_pairs.append((image_path, caption))
                        logger.debug(f"Added pair: ({image_path}, {caption})")
            else:
                missing_images_count += 1
                logger.warning(f"Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, caption = self.data_pairs[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption

# Image transformations for CLIP
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# Dataset and DataLoader
dataset = GQACLIPDataset(annotations, image_folder_path, transform=clip_transform)
logger.info(f"Dataset size: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
logger.info("DataLoader initialized.")

# Example: Displaying some data
for idx, (images, captions) in enumerate(dataloader):
    logger.info(f"\nBatch {idx + 1} loaded.")
    logger.info(f"  Image batch shape: {images.shape}")
    logger.info(f"  Captions (first 2): {captions[:2]}")
    logger.info("  ---")
    break  # To only show the first batch

