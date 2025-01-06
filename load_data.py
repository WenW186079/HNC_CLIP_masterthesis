import os
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import clip

# logger 
logger = logging.getLogger("HNC_CLIP_Logger")
logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class HNCCLIPDataset(Dataset):
    def __init__(self, annotations, image_folder, transform=None):
        """
        Initializes the dataset by creating image-caption pairs (only HNC negatives).
        """
        self.annotations = annotations
        self.image_folder = image_folder
        self.transform = transform
        self.data_pairs = [] # Store (image, positive_caption, negative_caption)

        self.hnc_count = 0

        logger.info("Creating image-caption pairs...")
        missing_images_count = 0

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

            else:
                missing_images_count += 1
                logger.warning(f"Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")
        logger.info(f"Total HNC pairs: {self.hnc_count}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption, source = self.data_pairs[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, pos_caption, neg_caption, image_path 

def fine_tune_collate_fn(batch):
    """
    Custom collate function to create in-batch negatives and log the pair count.
    """
    images, pos_captions, hnc_neg_captions = [], [], []

    for image, pos_caption, neg_caption, *_ in batch:
        images.append(image)
        pos_captions.append(pos_caption)
        hnc_neg_captions.append(neg_caption)

    tokenized_pos_captions = clip.tokenize(pos_captions)
    tokenized_hnc_neg_captions = clip.tokenize(hnc_neg_captions)

    # In-batch negatives: positive captions from other images
    in_batch_neg_captions = []
    total_pairs_per_sample = 0

    for i in range(len(pos_captions)):
        # Add all positive captions from the rest of the batch as negatives
        neg_captions_for_sample = [pos_captions[j] for j in range(len(pos_captions)) if j != i]
        in_batch_neg_captions.append(clip.tokenize(neg_captions_for_sample))
        
        num_pairs_for_sample = 1 + len(neg_captions_for_sample)  # 1 HNC negative + in-batch negatives
        total_pairs_per_sample += num_pairs_for_sample
        print(f"Sample {i + 1}: HNC negative + {len(neg_captions_for_sample)} in-batch negatives (Total pairs: {num_pairs_for_sample})")

    total_pairs = total_pairs_per_sample * len(batch)
    print(f"Total pairs in batch: {total_pairs}")

    return (
        torch.stack(images),
        tokenized_pos_captions,
        tokenized_hnc_neg_captions,
        in_batch_neg_captions
    )


def load_data_pairs(json_file_path, image_folder_path, batch_size=32, num_random_negatives=5, shuffle=True):
    """
    Creates a DataLoader for fine-tuning CLIP.

    """
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
    dataset = HNCCLIPDataset(annotations, image_folder_path, transform=clip_transform)
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=fine_tune_collate_fn)
    logger.info("DataLoader for fine-tuning initialized.")
    return dataloader
