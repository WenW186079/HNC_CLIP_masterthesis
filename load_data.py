import os
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
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

class HNCCLIPDataset(Dataset):
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
        self.random_negative_count = 0

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
        logger.info(f"Total HNC pairs: {self.hnc_count}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption, source = self.data_pairs[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, pos_caption, neg_caption, image_path 
    
    def get_unique_batch(self, batch_size):
        """
        Ensures no repeated image_path in a batch.
        """
        unique_paths = set()
        unique_batch = []

        for data_pair in self.data_pairs:
            image_path = data_pair[0]
            if image_path not in unique_paths:
                unique_paths.add(image_path)
                unique_batch.append(data_pair)

            if len(unique_batch) == batch_size:
                break

        if len(unique_batch) < batch_size:
            raise ValueError(f"❗️Not enough unique image paths to form a batch of size {batch_size}")

        return unique_batch
    
    def pair_data_tensor_unique(batch, tokenizer):
        """
        Create tensor pairs for (image, positive_caption, hard_negative_caption),
        ensuring no duplicate images in a batch.
        """
        images, pos_captions, neg_captions, image_paths = batch
        batch_size = len(images)
        
        print(f"Batch size: {batch_size}")

        # Tokenize captions
        pos_captions = [tokenizer(caption, truncate=True).to(images[0].device) for caption in pos_captions]
        neg_captions = [tokenizer(caption, truncate=True).to(images[0].device) for caption in neg_captions]

        # Collect unique image pairs
        paired_data = []
        for i in range(batch_size):
            paired_data.append((images[i], pos_captions[i], neg_captions[i]))

        return paired_data
        
    
    def pair_data(batch):
        """
        Create data pairs for training within a batch:
        (I_i, T_i), (I_i, T_i_HNC), (I_i, T_j), (I_i, T_j_HNC).
        """
        paired_data = []
        images, pos_captions, neg_captions, image_paths = batch
        batch_size = len(images)
        print('batch_size=',batch_size)


        for i in range(batch_size):
            image_tensor = images[i]
            pos_caption = pos_captions[i]
            neg_caption = neg_captions[i]

            # Add current image pairs
            paired_data.append((image_tensor, pos_caption, "positive"))  # Positive pair
            paired_data.append((image_tensor, neg_caption, "hnc"))       # HNC pair

            # Add cross-image negatives for all other images
            for j in range(batch_size):
                if j != i:
                    cross_pos_caption = pos_captions[j]
                    cross_neg_caption = neg_captions[j]
                    paired_data.append((image_tensor, cross_pos_caption, "random_negative"))  # Cross-image positive pair
                    paired_data.append((image_tensor, cross_neg_caption, "random_negative"))  # Cross-image HNC pair
        
        # Validation: Check if total pairs = 2n^2
        expected_pairs = 2 * batch_size ** 2
        if len(paired_data) != expected_pairs:
            print(f"❗️Error in pair count! Expected {expected_pairs}, but got {len(paired_data)}.")
            print(f"Batch size: {batch_size}")
            print(f"Generated pairs: {paired_data}")
        else:
            print(f"✅ Pair count is correct: {len(paired_data)} pairs (batch size: {batch_size}).")

        return paired_data

    def pair_data_tensor(batch, tokenizer):
        
        paired_data = []
        images, pos_captions, neg_captions, image_paths = batch 
        batch_size = len(images)
        print('batch_size=',batch_size)

        for i in range(batch_size):
            image_tensor = images[i]
            pos_caption = tokenizer(pos_captions[i], truncate=True).to(image_tensor.device)  # Tokenize positive caption
            neg_caption = tokenizer(neg_captions[i], truncate=True).to(image_tensor.device)  # Tokenize negative caption

            # Add current image pairs
            paired_data.append((image_tensor, pos_caption, "positive"))  # Positive pair
            paired_data.append((image_tensor, neg_caption, "hnc"))       # HNC pair

            # Add cross-image negatives for all other images
            for j in range(batch_size):
                if j != i:
                    cross_pos_caption = tokenizer(pos_captions[j], truncate=True).to(image_tensor.device)
                    cross_neg_caption = tokenizer(neg_captions[j], truncate=True).to(image_tensor.device)
                    paired_data.append((image_tensor, cross_pos_caption, "random_negative"))  # Cross-image positive pair
                    paired_data.append((image_tensor, cross_neg_caption, "random_negative"))  # Cross-image HNC pair

        # Validation: Check if total pairs = 2n^2
        expected_pairs = 2 * batch_size ** 2
        if len(paired_data) != expected_pairs:
            print(f"❗️ Error in pair count! Expected {expected_pairs}, but got {len(paired_data)}.")
            print(f"Batch size: {batch_size}")
            print(f"Generated pairs: {paired_data}")
        else:
            print(f"✅ Pair count is correct: {len(paired_data)} pairs (batch size: {batch_size}).")

        return paired_data

# Example usage:
train_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_train_sampled_1_percent.json'
val_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_val_sampled_1_percent.json'
image_folder_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images'

# Initialize CLIP Models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
tokenizer = clip.tokenize

# Load the annotations
with open(train_json_file_path, 'r') as f:
    train_annotations = json.load(f)

# Initialize the dataset
dataset = HNCCLIPDataset(
    annotations=train_annotations,
    image_folder=image_folder_path,
    transform=preprocess, 
)

# Create a DataLoader
batch_size = 3 
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print the first batch
for batch in data_loader:
    images, pos_captions, neg_captions, image_paths = batch
    for i in range(len(images)):
        print(f"Image Path: {image_paths[i]}")
        print(f"Positive Caption: {pos_captions[i]}")
        print(f"Negative Caption: {neg_captions[i]}")
        print("-" * 50)
    break

for batch in data_loader:
    paired_batch = HNCCLIPDataset.pair_data_tensor_unique(batch, tokenizer)
    # for i, (image_tensor, text_tensor, pair_type) in enumerate(paired_batch[:3]):
    for i, (image_tensor, pos_tensor, neg_caption) in enumerate(paired_batch):
        print(f"Pair {i+1}:")
        print(f"Image Tensor Shape: {image_tensor.shape}")  
        print(f"pos_tensor Tensor Shape: {pos_tensor.shape}")    
        print(f"neg_caption Tensor Shape: {neg_caption.shape}")    
        print("-" * 50)
    break
