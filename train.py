import os
import json
import random
from PIL import Image  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
import logging


from load_data import HNCCLIPDataset, load_data_pairs 


# Paths
json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_clean_strict_val.json'
image_folder_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images'

batch_size = 32
num_random_negatives = 5

# Create DataLoader
trai_data = load_data_pairs(
    json_file_path=json_file_path,
    image_folder_path=image_folder_path,
    batch_size=batch_size,
    num_random_negatives=num_random_negatives
)

print('finish')

# Display top 5 pairs and random 5 pairs
for batch_idx, (images, pos_captions, neg_captions, sources, image_paths) in enumerate(trai_data):
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