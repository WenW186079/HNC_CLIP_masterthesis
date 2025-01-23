import os
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import random
import clip

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class LoadHNCPair(Dataset):
    def __init__(self, annotations, image_folder, transform=None):
        """
        Initializes the dataset by creating (image, positive_caption, negative_caption) pairs.
        """
        self.annotations = annotations
        self.image_folder = image_folder
        self.transform = transform
        self.data_pairs = [] 
        self.hnc_count = 0
        self.positive_count = 0

        logger.info("Creating image-POS-HNC pairs...")
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

                self.positive_count += len(pos_caption_map)

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
                logger.warning(f"❗️Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")
        logger.info(f"Total Pos sampels: {self.positive_count}")
        logger.info(f"Total HNC sampels: {self.hnc_count}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption, source = self.data_pairs[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, pos_caption, neg_caption, image_path 
    
class UniqueImageSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        # Shuffle indices
        random.shuffle(self.indices)
        used_image_paths = set()

        batch = []
        for idx in self.indices:
            image_path, _, _, _ = self.dataset[idx]
            if image_path not in used_image_paths:
                batch.append(idx)
                used_image_paths.add(image_path)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    used_image_paths.clear()  

        # Yield the last batch if it's not empty
        if batch:
            yield batch

    def __len__(self):
        return len(self.indices) // self.batch_size


def show_batches(data_loader):
    """
    Print the dataset of the first batch, checking for repeated images.

    """
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx == 0:
            print(f"\n[Batch {batch_idx}] Dataset:")
            for i in range(len(batch[3])): 
                image = batch[0][i]
                image_path = batch[3][i]
                pos_caption = batch[1][i]
                neg_caption = batch[2][i]

                print(f"Image: {image}")
                print(f"Image.shape: {image.shape}")
                print(f"Image Path: {image_path}")
                print(f"  Positive Caption: {pos_caption}")
                print(f"  Negative Caption: {neg_caption}")
            break

# # Example usage:
# train_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_train_sampled_1_percent.json'
# val_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_val_sampled_1_percent.json'
# image_folder_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images'

# # Initialize CLIP Models
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/32", device=device)
# tokenizer = clip.tokenize

# # Load the annotations
# with open(train_json_file_path, 'r') as f:
#     train_annotations = json.load(f)

# # Initialize the dataset
# dataset = LoadHNCPair(
#     annotations=train_annotations,
#     image_folder=image_folder_path,
#     transform=preprocess, 
# )

# # Create a DataLoader
# batch_size = 3 
# sampler = UniqueImageSampler(dataset, batch_size)
# data_loader = DataLoader(dataset, batch_sampler=sampler)
# show_batches(data_loader)
