import os
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import argparse
import re 
import numpy as np

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

###############################################################################
# 数据集：同时对图像和文本进行预处理，输出 tensor
###############################################################################
class COCORetrievalDataset(Dataset):
    """
    该数据集类同时对图像和文本进行处理，输出的结果都是 tensor，
    可直接用于后续余弦相似度计算。
    
    参数：
      - root_dir: 数据集根目录（应包含标注文件和图片文件夹）
      - split: 'val' 或 'test'
      - max_words: 对 caption 进行截断时的最大词数
      - image_preprocess: 对 PIL Image 进行处理，返回 tensor（例如，resize、ToTensor、normalize）
      - text_preprocess: 对文本字符串进行处理，返回 tensor（例如，tokenize 后取 embedding）
    """
    def __init__(self, root_dir=COCO_ROOT, split="test", max_words=30, image_preprocess=None, text_preprocess=None):
        filenames = {
            'val': 'coco_karpathy_val.json',
            'test': 'coco_karpathy_test.json'
        }
        json_path = os.path.join(root_dir, filenames[split])
        self.annotation = json.load(open(json_path, 'r'))
        self.root_dir = root_dir
        self.image_preprocess = image_preprocess
        self.text_preprocess = text_preprocess

        # 构建图像文件列表和文本列表，并建立映射关系
        self.image_files = []
        self.texts = []
        self.img2txt = {}  # key: image index, value: list of caption indices
        self.txt2img = {}  # key: caption index, value: image index

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image_files.append(ann['image'])
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                processed_caption = pre_caption(caption, max_words)
                if self.text_preprocess is not None:
                    # 调用 text_preprocess 得到 tensor（例如 embedding）
                    text_tensor = self.text_preprocess(processed_caption)
                else:
                    # 若没有提供 text_preprocess，则直接返回原始字符串（不推荐用于余弦相似度计算）
                    text_tensor = processed_caption
                self.texts.append(text_tensor)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # 加载并预处理图像
        image_path = os.path.join(self.root_dir, "val2014", self.image_files[index])
        image = Image.open(image_path).convert("RGB")
        if self.image_preprocess is not None:
            image_tensor = self.image_preprocess(image)
        else:
            # 默认转换为 tensor
            image_tensor = transforms.ToTensor()(image)
        # 获取该图像对应的所有文本 tensor（列表形式）
        caption_indices = self.img2txt[index]
        text_tensors = [self.texts[i] for i in caption_indices]
        return {"image": image_tensor, "texts": text_tensors, "idx": index}

###############################################################################
# 检索指标计算函数
###############################################################################
def compute_text_retrieval(scores, img2txt):
    prec_at_1 = AverageMeter()
    prec_at_5 = AverageMeter()
    for i in tqdm(range(scores.shape[0]), desc="Text Retrieval"):
        top5_captions = np.argsort(-scores[i])[:5]
        true_captions = img2txt[i]
        prec_at_1.update(top5_captions[0] in true_captions)
        prec_at_5.update(any(x in true_captions for x in top5_captions))
    return {"TextPrec@1": prec_at_1.avg, "TextPrec@5": prec_at_5.avg}

def compute_image_retrieval(scores, txt2img):
    image_prec_at_1 = AverageMeter()
    image_prec_at_5 = AverageMeter()
    for i in tqdm(range(scores.shape[1]), desc="Image Retrieval"):
        top5_images = np.argsort(-scores[:, i])[:5]
        true_image = txt2img[i]
        image_prec_at_1.update(true_image == top5_images[0])
        image_prec_at_5.update(true_image in top5_images)
    return {"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg}

def compute_retrieval_metrics(scores, img2txt, txt2img):
    text_metrics = compute_text_retrieval(scores, img2txt)
    image_metrics = compute_image_retrieval(scores, txt2img)
    return {**text_metrics, **image_metrics}

###############################################################################
# 示例运行代码
###############################################################################
if __name__ == "__main__":
    # 定义图像预处理函数：调整大小并转换为 tensor
    image_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 如有需要，可添加 normalize 等操作
    ])

    # 定义文本预处理函数：这里使用一个 dummy 实现，实际中请替换为基于模型的文本编码
    def dummy_text_preprocess(text):
        # 例如，使用 CLIPTokenizer 和 CLIPModel.get_text_features 计算文本 embedding
        # 这里返回一个固定维度随机 tensor 模拟文本 embedding，维度为 512
        return torch.randn(512)

    # 构造数据集
    dataset = COCORetrievalDataset(
        root_dir=COCO_ROOT,
        split="val",
        max_words=30,
        image_preprocess=image_preprocess,
        text_preprocess=dummy_text_preprocess
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 说明：
    # 此处假设 dataset 输出的 "image" 已经是模型输入格式的 tensor，
    # "texts" 则为列表，每个元素是文本 embedding 的 tensor。
    # 接下来你可以直接用这些 tensor 计算余弦相似度。
    #
    # 例如，这里简单模拟将 image tensor 当作 image feature，
    # 文本 embedding已经由 dummy_text_preprocess 得到，
    # 然后归一化后计算内积模拟余弦相似度。

    all_image_features = []
    all_text_features = []
    # 注意：img2txt 和 txt2img 映射关系在 dataset 中保存
    img2txt = dataset.img2txt
    txt2img = dataset.txt2img

    for batch in tqdm(dataloader, desc="Collecting features"):
        # batch["image"]: shape (B, C, H, W) —— 这里假定已是特征向量，如果实际需要再送入模型，请在此处调用模型
        all_image_features.append(batch["image"])
        # 对于文本，每个 batch 中每个样本包含多个文本 embedding，逐个加入列表
        for text_list in batch["texts"]:
            for t in text_list:
                all_text_features.append(t.unsqueeze(0))  # 增加 batch 维度

    all_image_features = torch.cat(all_image_features, dim=0)  # shape: (N_images, C, H, W) or (N_images, D) 如果已提取特征
    all_text_features = torch.cat(all_text_features, dim=0)      # shape: (N_text, D)

    # 为了计算余弦相似度，我们假设 all_image_features 和 all_text_features 均为一维 embedding（D维）
    # 如果 image tensor 仍为 (B, C, H, W)，请在此处使用模型进一步提取特征
    # 此处仅做示例，将其视为 D 维向量
    all_image_features = all_image_features.view(all_image_features.size(0), -1)
    all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

    scores = (all_image_features @ all_text_features.T).numpy()

    results = compute_retrieval_metrics(scores, img2txt, txt2img)
    print("检索指标结果：", results)

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate Retrieval CLIP Model")
#     parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file.")
#     parser.add_argument("--image_folder", type=str, required=True, help="Path to the image folder.")
#     parser.add_argument("--model_name", type=str, required=True, help="Name of the CLIP model to use.")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
#     parser.add_argument("--dataset_type", type=str, required=True, choices=["hnc", "coco"], help="Type of dataset (hnc or coco).")

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CLIPModel.from_pretrained(args.model_name).to(device)
#     processor = CLIPProcessor.from_pretrained(args.model_name)

#     if args.dataset_type == "hnc":
#         eval_dataset = LoadHNCPair(annotations=json.load(open(args.dataset)), image_folder=args.image_folder)
#     elif args.dataset_type == "coco":
#         eval_dataset = LoadCOCOPair(annotations=json.load(open(args.dataset)), image_folder=args.image_folder)

#     text_recall_1, text_recall_5, image_recall_1, image_recall_5 = evaluate_retrieval(
#         model, processor, eval_dataset, device, batch_size=args.batch_size
#     )

#     print(f"Text Recall@1: {text_recall_1 * 100:.2f}%")
#     print(f"Text Recall@5: {text_recall_5 * 100:.2f}%")
#     print(f"Image Recall@1: {image_recall_1 * 100:.2f}%")
#     print(f"Image Recall@5: {image_recall_5 * 100:.2f}%")

# if __name__ == "__main__":
#     main()
