import os
import re
import json
import numpy as np

from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from transformers import CLIPProcessor, CLIPModel

COCO_ROOT = os.path.expanduser("./Coco")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    
    return caption

filenames = {
    'val': 'coco_karpathy_val.json',
    'test': 'coco_karpathy_test.json'
}

class COCO_Retrieval(Dataset):
    def __init__(self, image_preprocess=None, root_dir=COCO_ROOT, max_words=30, split="test", download=False):  
        """
        COCO Retrieval Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: image perturbation function for patch permutation experiments.
        download: Whether to download the dataset if it does not exist.
        """
        self.root_dir = root_dir
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.image_root = root_dir
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, "val2014", self.annotation[index]['image'])   
        image = Image.open(image_path).convert('RGB')    
        
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)
         
        return {"image": image, "idx": index}
        
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
    
        else:
            scores_t2i = scores
            scores_i2t = scores

        print(f"COCO results across {scores_i2t.shape} samples. ")
        prec_at_1 = AverageMeter()
        prec_at_5 = AverageMeter()

        # Text retrieval
        tqdm_iterator = tqdm(range(len(self.img2txt)))
        for i in tqdm_iterator:
            top5_captions = np.argsort(scores_i2t[i])[-5:]
            true_captions = self.img2txt[i]

            prec_at_1.update(len(set(true_captions) & set(top5_captions[-1:]))>0)
            prec_at_5.update(len(set(true_captions) & set(top5_captions))>0)

            tqdm_iterator.set_description(f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f}")

        # Image Retrieval
        image_prec_at_1 = AverageMeter()
        image_prec_at_5 = AverageMeter()

        tqdm_iterator = tqdm(range(len(self.txt2img)))
        for i in tqdm_iterator:
            top5_images = np.argsort(scores_t2i[:, i])[-5:]
            true_image = self.txt2img[i]

            image_prec_at_1.update(true_image in top5_images[-1:])
            image_prec_at_5.update(true_image in top5_images)

            tqdm_iterator.set_description(f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f}")

        records = [{"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg, "TextPrec@1": prec_at_1.avg, "TextPrec@5": prec_at_5.avg}]
        return records


###############################################################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "openai/clip-vit-base-patch32"  
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    def clip_image_preprocess(image):
        processed = processor(images=image, return_tensors="pt")["pixel_values"]
        return processed.squeeze(0)

    dataset = COCO_Retrieval(image_preprocess=clip_image_preprocess, root_dir=COCO_ROOT, split="test")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_image_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image features"):
            # batch["image"] 的 shape 为 (B, 3, 224, 224)
            images = batch["image"].to(device)
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu())
    all_image_features = torch.cat(all_image_features, dim=0)  # shape: (N_images, D)

    # （6）提取文本特征：利用 COCO_Retrieval 中保存的文本
    all_texts = dataset.text  # 这是一个列表，每个元素是预处理后的 caption（字符串）
    all_text_features = []
    batch_size = 32
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Extracting text features"):
        batch_texts = all_texts[i:i+batch_size]
        # 使用 processor 对文本进行 tokenize
        text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        # 归一化文本特征
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_text_features.append(text_features.cpu())
    all_text_features = torch.cat(all_text_features, dim=0)  # shape: (N_text, D)

    scores = (all_image_features @ all_text_features.T).numpy()  # shape: (N_images, N_text)

    results = dataset.evaluate_scores(scores)
    print("Evaluation Results:", results)