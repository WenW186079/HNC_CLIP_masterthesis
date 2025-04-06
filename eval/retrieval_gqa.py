import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# ✅ 你的数据集路径
IMAGE_ROOT = os.path.expanduser("./gqa_dataset/images/images")  # 图片路径
CAPTION_FILE = os.path.expanduser("./HNC/hnc_val_sampled_1_percent.json")  # Caption JSON 路径

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
        
def pre_caption(caption, max_words=50):
    """清理并截断 caption"""
    caption = caption.lower().replace("\n", " ").strip()
    return " ".join(caption.split()[:max_words])  # 限制单词数

class GQA_Retrieval(Dataset):
    def __init__(self, image_preprocess=None, max_words=30):
        """
        GQA 数据集用于 Image-Text Retrieval
        - image_preprocess: 图片预处理函数
        - max_words: 最大单词数
        """
        self.image_preprocess = image_preprocess

        # ✅ 读取 GQA caption JSON 文件
        with open(CAPTION_FILE, 'r') as f:
            self.annotation = json.load(f)

        self.text = []  # 存储所有 captions（包含正负样本）
        self.image = []  # 存储所有图像路径
        self.txt2img = {}  # Mapping: Text → Image
        self.img2txt = {}  # Mapping: Image → List[Text]（仅存储正样本）

        txt_id = 0
        for image_file, data in self.annotation.items():
            image_path = os.path.join(IMAGE_ROOT, image_file + ".jpg")  # ✅ 正确拼接图片路径
            self.image.append(image_path)
            self.img2txt[len(self.image) - 1] = []  # **仅存储正样本**

            for caption_id, caption_data in data["captions"].items():
                caption = pre_caption(caption_data["caption"], max_words)
                label = caption_data["label"]  # 1 = 正样本, 0 = 负样本

                self.text.append(caption)  # 存储所有 caption
                self.txt2img[txt_id] = len(self.image) - 1  # Text → Image

                if label == 1:  # **只有正样本存入 img2txt**
                    self.img2txt[len(self.image) - 1].append(txt_id)

                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        image = Image.open(image_path).convert('RGB')

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        return {"image": image, "idx": index}

    def evaluate_scores(self, scores):
        """
        计算：
        - Text Retrieval: 给定图像，找到正确的 caption
        - Image Retrieval: 给定 caption，找到正确的图像
        """
        scores_i2t, scores_t2i = scores  # Image-to-Text, Text-to-Image

        print(f"GQA Retrieval Results across {scores_i2t.shape} samples.")
        prec_at_1, prec_at_5 = AverageMeter(), AverageMeter()
        image_prec_at_1, image_prec_at_5 = AverageMeter(), AverageMeter()

        # **Text Retrieval（给定图像，找到正确的 caption）**
        for i in tqdm(range(len(self.img2txt)), desc="Evaluating Text Retrieval"):
            top5_captions = np.argsort(scores_i2t[i])[-5:]
            true_captions = self.img2txt[i]  # **仅包括正样本的索引**

            prec_at_1.update(len(set(true_captions) & set(top5_captions[-1:])) > 0)
            prec_at_5.update(len(set(true_captions) & set(top5_captions)) > 0)

        # **Image Retrieval（给定 caption，找到正确的图像）**
        for i in tqdm(range(len(self.txt2img)), desc="Evaluating Image Retrieval"):
            if i not in self.img2txt or len(self.img2txt[self.txt2img[i]]) == 0:  # ✅ **如果 caption 是负样本，不计算**
                continue
            
            top5_images = np.argsort(scores_t2i[:, i])[-5:]
            true_image = self.txt2img[i]

            image_prec_at_1.update(true_image in top5_images[-1:])
            image_prec_at_5.update(true_image in top5_images)

        results = {
            "Text Retrieval Prec@1": prec_at_1.avg,
            "Text Retrieval Prec@5": prec_at_5.avg,
            "Image Retrieval Prec@1": image_prec_at_1.avg,
            "Image Retrieval Prec@5": image_prec_at_5.avg
        }
        return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 CLIP Model
    # model_name = "openai/clip-vit-base-patch32"
    model_name = "WenWW/HNC_1_2048_epoch18" 
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    def clip_image_preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    # 加载 GQA 数据集
    dataset = GQA_Retrieval(image_preprocess=clip_image_preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 提取图片特征
    all_image_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image features"):
            images = batch["image"].to(device)
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu())

    all_image_features = torch.cat(all_image_features, dim=0)

    # 提取文本特征
    all_texts = dataset.text
    all_text_features = []
    batch_size = 32
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Extracting text features"):
        batch_texts = all_texts[i:i+batch_size]
        text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_text_features.append(text_features.cpu())

    all_text_features = torch.cat(all_text_features, dim=0)

    # 计算相似度得分
    scores_i2t = (all_image_features @ all_text_features.T).numpy()
    scores_t2i = scores_i2t.T  

    # 评估
    results = dataset.evaluate_scores((scores_i2t, scores_t2i))
    print("Evaluation Results:", results)
