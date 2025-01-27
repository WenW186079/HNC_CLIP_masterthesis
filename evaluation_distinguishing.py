import os
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader

from load_data import LoadHNCPair

batch_size = 32
# val_path="/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_val_sampled_1_percent.json"
# val_path="/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_clean_strict_val.json"
val_path="/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_val_sampled_10_percent.json"
image_folder_path = "/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images" 

# model_name="openai/clip-vit-base-patch32" 
model_name='Nano1337/openclip-negclip' 
# model_name= 'WenWW/HNC_CLIP_B32_1.0' 


def evaluate_hnc_clip(model, processor, dataset, device, batch_size=32):
    """
     The accuracy of the model in distinguishing positive from negative captions.
    """
    model.eval()
    total_samples = 0
    correct_predictions = 0

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            image_paths, pos_captions, neg_captions = zip(*batch)

            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

            # Randomly shuffle between positive and negative captions
            captions = []
            labels = [] 
            for pos, neg in zip(pos_captions, neg_captions):
                if random.random() < 0.5:
                    captions.append(pos)
                    labels.append(1)
                else:
                    captions.append(neg)
                    labels.append(0)


            text_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)

            image_features = model.get_image_features(**inputs)
            text_features = model.get_text_features(**text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = torch.matmul(image_features, text_features.T).diagonal()

            # Convert similarities to predictions
            predictions = (similarities > 0.5).long().tolist()

            # Compute accuracy
            correct_predictions += sum(p == l for p, l in zip(predictions, labels))
            total_samples += len(labels)

    accuracy = correct_predictions / total_samples
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    val_annotations = json.load(open(val_path))
    val_dataset = LoadHNCPair(annotations=val_annotations, image_folder=image_folder_path)

    accuracy = evaluate_hnc_clip(model, processor, val_dataset, device, batch_size=batch_size)
    print(f"Model {model_name}: Validation Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

