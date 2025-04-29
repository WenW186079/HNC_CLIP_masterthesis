import argparse
import os
import logging
import random
import numpy as np
import torch
import clip

from eval_functions import (evaluate_cosine_similarities,
                   evaluate_cosine_similarities_and_plot,
                   evaluate_cosine_similarities_random_negtive,
                   evaluate_caption_accuracy,
                   get_caption_ratios)

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from load_data import load_split


def load_finetuned_clip_model(model_name, checkpoint_path, device='cuda', finetune_mode=None):
    model, preprocess = clip.load(model_name, device=device)
    tokenizer = clip.tokenize

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"‼️⚠️Error loading checkpoint from '{checkpoint_path}': {e}")
            print("Please check that the checkpoint file is not corrupted and was saved properly.")
            raise e

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"‼️⚠️Warning: Missing keys when loading checkpoint from '{checkpoint_path}':\n", load_result.missing_keys)
        if load_result.unexpected_keys:
            print(f"‼️⚠️Warning: Unexpected keys when loading checkpoint from '{checkpoint_path}':\n", load_result.unexpected_keys)
        
        epoch = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
        saved_mode = checkpoint.get("finetune_mode", None) if isinstance(checkpoint, dict) else None
        if finetune_mode is not None and saved_mode is not None and saved_mode != finetune_mode:
            print(f"‼️⚠️Warning: Checkpoint was fine-tuned in mode '{saved_mode}', but '{finetune_mode}' was requested.")
        print(f"✅Loaded fine-tuned model from '{checkpoint_path}' (Epoch: {epoch}).")
    else:
        print(f"‼️⚠️Checkpoint not found at '{checkpoint_path}'. Using base model.")

    return model, preprocess, tokenizer

def set_determinism(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_comparison_table(results):
    if results and "Threshold" in results[0]:
        headers = ["Checkpoint", "Eval_Method", "Threshold", "Accuracy"]
        header_line = " | ".join(headers)
        print("\n" + header_line)
        print("-" * len(header_line))
        for r in results:
            row = [
                str(r.get("Checkpoint", "")),
                str(r.get("Eval_Method", "")),
                f"{r.get('Threshold', 0):.2f}",
                f"{r.get('Accuracy', 0):.8f}"
            ]
            print(" | ".join(row))
    else:
        headers = ["Checkpoint", "Eval_Method", "Avg_Pos", "Avg_Neg", "Margin"]
        if results and "Avg_Rand_Neg" in results[0]:
            headers.append("Avg_Rand_Neg")
        header_line = " | ".join(headers)
        print("\n" + header_line)
        print("-" * len(header_line))
        for r in results:
            row = [
                str(r.get("Checkpoint", "")),
                str(r.get("Eval_Method", "")),
                f"{r.get('Avg_Pos', 0):.8f}",
                f"{r.get('Avg_Neg', 0):.8f}",
                f"{r.get('Margin', 0):.8f}"
            ]
            if "Avg_Rand_Neg" in r:
                row.append(f"{r.get('Avg_Rand_Neg', 0):.8f}")
            print(" | ".join(row))
    print()

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP models using different evaluation methods")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "finetuned"],
                        help="Select 'base' for the original CLIP model, or 'finetuned' for saved checkpoint(s).")
    parser.add_argument("--checkpoint_path", type=str, nargs="+", default=["./checkpoints_dpo/final_model.pt"],
                        help="Path(s) to the fine-tuned model checkpoint(s) (used if model_type is 'finetuned').")
    parser.add_argument("--eval_method", type=str, default="random", choices=["cosine", "plot", "random",'distinguish'],
                        help="Evaluation method: 'cosine' for evaluate_cosine_similarities, 'plot' for evaluate_cosine_similarities_and_plot, and 'random' for evaluate_cosine_similarities_random_negtive.")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="Name of the CLIP model (e.g., ViT-B/32)")
    parser.add_argument("--test_json", type=str, default="./HNC/hnc_clean_strict_test.json", help="Path to test json file")
    parser.add_argument("--images_path", type=str, default="./gqa_dataset/images/images", help="Path to test images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--finetune_mode", type=str, default="full_encoder", choices=["text_encoder", "vision_encoder", "full_encoder"],
                        help="The fine-tuning mode that was applied when training the checkpoint.")
    parser.add_argument("--loader_type", type=str, default="hnc", choices=["hnc", "coco"],
                        help="The dataset type to load: 'hnc' for the standard HNC dataset, 'coco' for COCO-style paired data.")
    args = parser.parse_args()

    set_determinism(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = [] 
    
    checkpoint_paths = args.checkpoint_path if args.model_type == "finetuned" else [None]
    
    for cp_path in checkpoint_paths:
        if args.model_type == "finetuned" and cp_path is not None:
            model, preprocess, tokenizer = load_finetuned_clip_model(args.model_name, cp_path, device=device, finetune_mode=args.finetune_mode)
        else:
            model, preprocess = clip.load(args.model_name, device=device)
            tokenizer = clip.tokenize

        model.eval()
        test_loader, test_dataset = load_split(args.test_json, "test", args.images_path, tokenizer, preprocess, args.batch_size, subset_size=None, loader_type=args.loader_type)
        print(f"Test Dataset size: {len(test_dataset)}")
        
        eval_method = args.eval_method
        if eval_method == "cosine":
            with torch.no_grad():
                avg_pos, avg_neg, margin = evaluate_cosine_similarities(model, test_loader, device)
            print(f"Checkpoint: {cp_path if cp_path else 'base'} | Eval: cosine")
            print(f"  Avg_Pos: {avg_pos:.8f}, Avg_Neg: {avg_neg:.8f}, Margin: {margin:.8f}")
            results.append({
                "Checkpoint": cp_path if cp_path else "base",
                "Eval_Method": "cosine",
                "Avg_Pos": avg_pos,
                "Avg_Neg": avg_neg,
                "Margin": margin
            })
        elif eval_method == "plot":
            plot_filename = os.path.basename(cp_path) if cp_path else "base"
            avg_pos, avg_neg, margin, _ = evaluate_cosine_similarities_and_plot(
                model,
                test_loader,
                device,
                plot_title=f"Image vs. Text Cosine Similarity for {plot_filename}",
                figsize=(12, 6),
                save_plot=True,
                plot_path=f"combined_similarity_matrix_{plot_filename}.png"
            )
            print(f"Checkpoint: {cp_path if cp_path else 'base'} | Eval: plot")
            print(f"  Avg_Pos: {avg_pos:.8f}, Avg_Neg: {avg_neg:.8f}, Margin: {margin:.8f}")
            results.append({
                "Checkpoint": cp_path if cp_path else "base",
                "Eval_Method": "plot",
                "Avg_Pos": avg_pos,
                "Avg_Neg": avg_neg,
                "Margin": margin
            })
        elif eval_method == "random":
            avg_pos, avg_neg, avg_rand_neg, margin = evaluate_cosine_similarities_random_negtive(model, test_loader, device)
            print(f"Checkpoint: {cp_path if cp_path else 'base'} | Eval: random")
            print(f"  Avg_Pos: {avg_pos:.8f}, Avg_Neg: {avg_neg:.8f}, Avg_Rand_Neg: {avg_rand_neg:.8f}, Margin: {margin:.8f}")
            results.append({
                "Checkpoint": cp_path if cp_path else "base",
                "Eval_Method": "random",
                "Avg_Pos": avg_pos,
                "Avg_Neg": avg_neg,
                "Margin": margin,
                "Avg_Rand_Neg": avg_rand_neg
            })
        elif eval_method == "distinguish":
            thresholds = [1, 1.1, 1.2, 1.5, 2, 3]
            ratios = get_caption_ratios(model, test_loader, device)
            total_samples = len(ratios)
            for th in thresholds:
                num_correct = sum(1 for r in ratios if r >= th)
                accuracy = num_correct / total_samples if total_samples > 0 else 0
                print(f"Checkpoint: {cp_path if cp_path else 'base'} | Eval: distinguish | Threshold: {th:.2f}")
                print(f"  Accuracy: {accuracy:.8f}")
                results.append({
                    "Checkpoint": cp_path if cp_path else "base",
                    "Eval_Method": "distinguish",
                    "Threshold": th,
                    "Accuracy": accuracy,
                })
        else:
            print("Unknown evaluation method.")
    
    print("\nComparison Table:")
    print_comparison_table(results)

if __name__ == "__main__":
    main()
