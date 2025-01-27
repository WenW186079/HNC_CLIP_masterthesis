import os
import json
import random
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import argparse

from load_data import LoadHNCPair, LoadCOCOPair


def distinguish_clip(model, processor, dataset, device, batch_size=32):
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
    parser = argparse.ArgumentParser(description="Evaluate CLIP Model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the image folder.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the CLIP model to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["hnc", "coco"], help="Type of dataset (hnc or coco).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    eval_annotations = json.load(open(args.dataset))

    if args.dataset_type == "hnc":
        eval_dataset = LoadHNCPair(annotations=eval_annotations, image_folder=args.image_folder)
    elif args.dataset_type == "coco":
        eval_dataset = LoadCOCOPair(annotations=eval_annotations, image_folder=args.image_folder)

    accuracy = distinguish_clip(model, processor, eval_dataset, device, batch_size=args.batch_size)
    print(f"Model {args.model_name}: Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
