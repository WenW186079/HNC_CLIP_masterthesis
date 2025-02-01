import os
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd

from load_data import LoadHNCPair, LoadCOCOPair


def distinguish_clip(model, processor, dataset, device, batch_size=32):
    """
     The accuracy of the model in distinguishing positive from negative captions.
     
     Further: pos_similarities / (neg_similarities + 1e-6)  > 1.1

    """
    model.eval()
    total_samples = 0
    correct_predictions = 0

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            image_paths, pos_captions, neg_captions = zip(*batch)

            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

            pos_text_inputs = processor(text=pos_captions, return_tensors="pt", padding=True).to(device)
            neg_text_inputs = processor(text=neg_captions, return_tensors="pt", padding=True).to(device)

            image_features = model.get_image_features(**image_inputs)
            pos_text_features = model.get_text_features(**pos_text_inputs)
            neg_text_features = model.get_text_features(**neg_text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            pos_text_features = pos_text_features / pos_text_features.norm(dim=-1, keepdim=True)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            pos_similarities = (image_features * pos_text_features).sum(dim=-1)  
            neg_similarities = (image_features * neg_text_features).sum(dim=-1) 

            similarity_ratio = pos_similarities / (neg_similarities + 1e-6)
            correct_predictions += (similarity_ratio > 1.1).sum().item()
            
            # correct_predictions += (pos_similarities > neg_similarities).sum().item()
            total_samples += len(image_paths)

    accuracy = correct_predictions / total_samples
    return accuracy, total_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP Model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the image folder.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the CLIP model to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["hnc", "coco"], help="Type of dataset (hnc or coco).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    eval_annotations = json.load(open(args.dataset))

    if args.dataset_type == "hnc":
        eval_dataset = LoadHNCPair(annotations=eval_annotations, image_folder=args.image_folder)
    elif args.dataset_type == "coco":
        eval_dataset = LoadCOCOPair(annotations=eval_annotations, image_folder=args.image_folder)

    model_names = args.model_name.split(",")
    results = []
    for model_name in model_names:
        print(f"\nEvaluating Model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)

        accuracy, sample_numbers = distinguish_clip(model, processor, eval_dataset, device, batch_size=args.batch_size)
        print(f"\nAcc: {accuracy}")
        
        results.append({
            "Model Name": model_name,
            "Sample Numbers": sample_numbers,
            "Accuracy (%)": round(accuracy * 100, 2)
        })
    
    df = pd.DataFrame(results)
    print("\n=============Evaluation Results==============\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
