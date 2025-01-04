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
logger.info("Loading annotations JSON file...")
with open(json_file_path, 'r') as f:
    annotations = json.load(f)
logger.info(f"Loaded {len(annotations)} annotations.")


class GQACLIPDataset(Dataset):
    def __init__(self, annotations, image_folder, transform=None):
        self.annotations = annotations
        self.image_folder = image_folder
        self.transform = transform
        self.data_pairs = [] # Stores (image_path, positive_caption, negative_caption)

        # Create (image_path, caption) pairs
        logger.info("Creating image-caption pairs...")
        missing_images_count = 0
        
        # Iterate over the annotations (image IDs as keys)
        for img_id, data in annotations.items():
            # Convert image ID to image filename (e.g., "2386621" -> "2386621.jpg")
            image_filename = f"{img_id}.jpg"
            image_path = os.path.join(self.image_folder, image_filename)

            if os.path.exists(image_path):
                captions_dict = data.get("captions", {})
                positive_captions = [cap['caption'] for cap in captions_dict.values() if cap['label'] == 1]
                negative_captions = [cap['caption'] for cap in captions_dict.values() if cap['label'] == 0]

                if positive_captions and negative_captions:
                    for pos_caption in positive_captions:
                        for neg_caption in negative_captions:
                            # Pair each positive caption with a negative caption
                            self.data_pairs.append((image_path, pos_caption, neg_caption))
                            logger.debug(f"Added pair: ({image_path}, {pos_caption}, {neg_caption})")
            else:
                missing_images_count += 1
                logger.warning(f"Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption = self.data_pairs[idx] 
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, pos_caption, neg_caption  

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

# Display the first 5 (image, positive caption, negative caption) pairs
for idx, (images, pos_captions, neg_captions) in enumerate(dataloader):
    print(f"\n--- Batch {idx + 1} ---")
    for i in range(5):  # Print the first 5 pairs
        print(f"Pair {i+1}:")
        print(f"  Image shape: {images[i].shape}")  # Shape of the image tensor
        print(f"  Positive Caption: {pos_captions[i]}")
        print(f"  Negative Caption: {neg_captions[i]}")
        print("-" * 50)
    break  
