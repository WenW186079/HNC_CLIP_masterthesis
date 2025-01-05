import os
import json
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# logger 
logger = logging.getLogger("HNC_CLIP_Logger")
logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class HNCCLIPDataset(Dataset):
    def __init__(self, annotations, image_folder, transform=None, num_random_negatives=5):
        self.annotations = annotations
        self.image_folder = image_folder
        self.transform = transform
        self.num_random_negatives = num_random_negatives
        self.data_pairs = []  # Store (image, positive_caption, negative_caption, source)

        self.hnc_count = 0
        self.random_count = 0

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
                            self.hnc_count += 1

                        # Add N times random negatives
                        for _ in range(self.num_random_negatives):
                            while True:
                                random_image_path, random_caption = random.choice(all_positive_captions)
                                if random_image_path != image_path:  
                                    break 
                            self.data_pairs.append((image_path, pos_caption, random_caption, "random"))
                            self.random_count += 1   
            else:
                missing_images_count += 1
                logger.warning(f"Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")
        logger.info(f"Total HNC pairs: {self.hnc_count}")
        logger.info(f"Total Random pairs: {self.random_count}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption, source = self.data_pairs[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, pos_caption, neg_caption, source, image_path 

def load_data_pairs(json_file_path, image_folder_path, batch_size=32, num_random_negatives=5, shuffle=True):
    # Load annotations
    logger.info("Loading annotations JSON file...")
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    logger.info(f"Loaded {len(annotations)} annotations.")

    # Image transformations
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # Create dataset
    dataset = HNCCLIPDataset(annotations, image_folder_path, transform=clip_transform, num_random_negatives=num_random_negatives)
    logger.info(f"Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    logger.info("load_data_pairs initialized.")
    return dataloader
'''
# Paths
json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_clean_strict_val.json'
image_folder_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images'

# Create DataLoader
data_loader = load_data_pairs(json_file_path, image_folder_path, batch_size=32, num_random_negatives=5)

# Display top 5 pairs and random 5 pairs
for batch_idx, (images, pos_captions, neg_captions, sources, image_paths) in enumerate(data_loader):
    print(f"\nDisplay TOP 5 pairs: \n--- Batch {batch_idx + 1} ---")
    for i in range(min(5, len(image_paths))):  
        print(f"Pair {i + 1}:")
        print(f"  Image Path: {image_paths[i]}")
        print(f"  Positive Caption: {pos_captions[i]}")
        print(f"  Negative Caption: {neg_captions[i]}")
        print(f"  Source of Negative: {sources[i]}")
        print("-" * 50)

    print(f"\nDisplay random 5 pairs from each batch: \n--- Batch {batch_idx + 1} ---")
    random_indices = random.sample(range(len(image_paths)), min(5, len(image_paths)))
    
    for i, rand_idx in enumerate(random_indices):
        print(f"Pair {i + 1}:")
        print(f"  Image Path: {image_paths[rand_idx]}")
        print(f"  Positive Caption: {pos_captions[rand_idx]}")
        print(f"  Negative Caption: {neg_captions[rand_idx]}")
        print(f"  Source of Negative: {sources[rand_idx]}")
        print("-" * 50)

    break  
'''
