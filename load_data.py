import os
import logging
from torch.utils.data import Dataset, Sampler
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class LoadHNCPair(Dataset):
    def __init__(self, annotations, image_folder):
        """
        Initializes the dataset by creating (image_path, positive_caption, negative_caption) pairs.
        """
        self.annotations = annotations
        self.image_folder = image_folder
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
        
        return image_path, pos_caption, neg_caption
    

def show_batches(data_loader):
    """
    Print the dataset of the first batch, checking for repeated images.

    """
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx == 0:
            print(f"\n[Batch {batch_idx}] Dataset:")
            for i in range(len(batch[0])): 
                image_path = batch[0][i]
                pos_caption = batch[1][i]
                neg_caption = batch[2][i]

                print(f"Image Path: {image_path}")
                print(f"  Positive Caption: {pos_caption}")
                print(f"  Negative Caption: {neg_caption}")
            break



class LoadCOCOPair(Dataset):
    def __init__(self, annotations, image_folder):
        self.annotations = annotations
        self.image_folder = image_folder
        self.data_pairs = [] 
        self.neg_count = 0
        self.positive_count = 0

        logger.info("Creating COCO image-POS-NEG pairs...")
        missing_images_count = 0

        for data in annotations:
            image_filename = data.get("image")
            image_path = os.path.join(self.image_folder, image_filename)
            
            if os.path.exists(image_path):
                pos_caption = data.get("text")  
                neg_caption = data.get("neg_text")  
                
                if pos_caption and neg_caption:
                    self.data_pairs.append((image_path, pos_caption, neg_caption, "hnc"))
                    self.positive_count += 1
                    self.neg_count += 1
            else:
                missing_images_count += 1
                logger.warning(f"❗️Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")
        logger.info(f"Total Pos samples: {self.positive_count}")
        logger.info(f"Total COCONEG samples: {self.neg_count}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption, source = self.data_pairs[idx]
        
        return image_path, pos_caption, neg_caption
