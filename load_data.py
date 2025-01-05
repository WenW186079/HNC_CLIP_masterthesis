import os
import json
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


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
    def __init__(self, annotations, image_folder, transform=None, num_random_negatives=1):
        self.annotations = annotations
        self.image_folder = image_folder
        self.transform = transform
        self.num_random_negatives = num_random_negatives
        self.data_pairs = []  # Store (image, positive_caption, negative_caption, source)

        logger.info("Creating image-caption pairs...")
        missing_images_count = 0

        all_positive_captions = []  
        for img_id, data in annotations.items():
            image_filename = f"{img_id}.jpg"
            image_path = os.path.join(self.image_folder, image_filename)
            if os.path.exists(image_path):
                captions_dict = data.get("captions", {})
                for cap in captions_dict.values():
                    if cap["label"] == 1: 
                        all_positive_captions.append((image_path, cap["caption"]))

        for img_id, data in annotations.items():
            image_filename = f"{img_id}.jpg"
            image_path = os.path.join(self.image_folder, image_filename)
            
            if os.path.exists(image_path):
                captions_dict = data.get("captions", {})
                
                # Create a map from cap_id to caption data
                pos_caption_map = {
                    cap_id: cap_data["caption"]
                    for cap_id, cap_data in captions_dict.items() if cap_data["label"] == 1
                }

                for cap_id, cap_data in captions_dict.items():
                    if cap_data["label"] == 0: 
                        neg_caption = cap_data["caption"]
                        cpt_p_id = cap_data.get("cpt_p_id")
                        
                        if not cpt_p_id:
                            logger.warning(f"Missing cpt_p_id for cap_id: {cap_id} in {image_filename}. Skipping.")
                            continue
                        
                        cpt_p_id = str(cpt_p_id)

                        if cpt_p_id not in pos_caption_map:
                            logger.warning(f"Invalid cpt_p_id: {cpt_p_id} in image {image_filename}.")
                            logger.info(f"Negative Caption: {cap_data['caption']}")
                            logger.info(f"Available Positive cap_ids: {list(pos_caption_map.keys())}")
                        else:
                            pos_caption = pos_caption_map[cpt_p_id]
                            self.data_pairs.append((image_path, pos_caption, neg_caption, "hnc"))

                        # Add random negatives
                        for _ in range(self.num_random_negatives):
                            random_image_path, random_caption = random.choice(all_positive_captions)
                            if random_image_path != image_path:  
                                self.data_pairs.append((image_path, pos_caption, random_caption, "random"))
            else:
                missing_images_count += 1
                logger.warning(f"Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption, source = self.data_pairs[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, pos_caption, neg_caption, source


# Image transformations
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

# Display pairs
for idx, (images, pos_captions, neg_captions, sources) in enumerate(dataloader):
    print(f"\n--- Batch {idx + 1} ---")
    for i in range(5):
        print(f"Pair {i + 1}:")
        print(f"  Image Path: {dataset.data_pairs[i][0]}")
        print(f"  Positive Caption: {pos_captions[i]}")
        print(f"  Negative Caption: {neg_captions[i]}")
        print(f"  Source of Negative: {sources[i]}")  
        print("-" * 50)
    break


# Display random 5 pairs from each batch
for idx, (images, pos_captions, neg_captions, sources) in enumerate(dataloader):
    print(f"\n--- Batch {idx + 1} ---")

    batch_size = len(pos_captions)
    random_indices = random.sample(range(batch_size), min(5, batch_size))
    
    for i, rand_idx in enumerate(random_indices):
        print(f"Pair {i + 1}:")
        print(f"  Image Path: {dataset.data_pairs[rand_idx][0]}")
        print(f"  Positive Caption: {pos_captions[rand_idx]}")
        print(f"  Negative Caption: {neg_captions[rand_idx]}")
        print(f"  Source of Negative: {sources[rand_idx]}")  
        print("-" * 50)
    break
