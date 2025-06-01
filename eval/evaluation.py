import argparse
import os
import sys
import logging
import random
import numpy as np
import torch
import clip
import pandas as pd
import re

from eval_functions import (evaluate_cosine_similarities,
                   evaluate_cosine_similarities_and_plot,
                   evaluate_cosine_similarities_random_negative,
                   evaluate_thresholds_accuracy,
                   get_caption_ratios,
                   evaluate_random_and_thresholds
                   )

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from load_data import load_split


def load_finetuned_clip_model(model_name, checkpoint_path, device, finetune_mode=None):
    model, preprocess = clip.load(model_name, device=device)
    model = model.to(device)

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

    return model, preprocess

def format_model_name(path: str) -> str:
        if path is None or path == 'base':
            return 'base'
        parent = os.path.basename(os.path.dirname(path))
        fname = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r'epoch_(\d+)', fname)
        epoch = m.group(1) if m else ''
        return f"{parent}_epoch{epoch}" if epoch else parent

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark   = False


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP models using different evaluation methods")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "finetuned"],
                        help="Select 'base' for the original CLIP model, or 'finetuned' for saved checkpoint(s).")
    parser.add_argument("--checkpoint_path", type=str, nargs="+", default=["./checkpoints_dpo/final_model.pt"],
                        help="Path(s) to the fine-tuned model checkpoint(s) (used if model_type is 'finetuned').")
    parser.add_argument("--eval_method", type=str, default="random", choices=["cosine", "plot", "random",'thresholds','random+thresholds'],
                        help="Evaluation method: 'cosine' for evaluate_cosine_similarities, 'plot' for evaluate_cosine_similarities_and_plot, and 'random' for evaluate_cosine_similarities_random_negative.")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="Name of the CLIP model (e.g., ViT-B/32)")
    parser.add_argument("--test_json", type=str, default="./HNC/hnc_clean_strict_test.json", help="Path to test json file")
    parser.add_argument("--images_path", type=str, default="./gqa_dataset/images/images", help="Path to test images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--finetune_mode", type=str, default="full_encoder", choices=["text_encoder", "vision_encoder", "full_encoder", 'last_encoder'],
                        help="The fine-tuning mode that was applied when training the checkpoint.")
    parser.add_argument("--loader_type", type=str, default="hnc", choices=["hnc", "coco"],
                        help="The dataset type to load: 'hnc' for the standard HNC dataset, 'coco' for COCO-style paired data.")
    parser.add_argument("--output_csv", type=str, default="result_scores.csv",
                        help="Path to write the results CSV file")
    args = parser.parse_args()

    set_all_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        logging.warning("‼️⚠️CUDA not available — running on CPU.")

    _ , preprocess = clip.load("ViT-B/32", device=device)
    tokenizer = clip.tokenize

    test_loader, test_dataset = load_split(
        args.test_json,
        'test',
        args.images_path,
        tokenizer,
        preprocess,
        args.batch_size,
        subset_size=1000,
        loader_type=args.loader_type
    )
    print(f"Test Dataset size: {len(test_dataset)}")

    results = [] 
    
    checkpoint_paths = args.checkpoint_path if args.model_type == "finetuned" else [None]
    thresholds = [1, 1.1, 1.2, 1.5, 2, 3]

    for cp_path in checkpoint_paths:
        if args.model_type == "finetuned" and cp_path is not None:
            model, preprocess = load_finetuned_clip_model(args.model_name, cp_path, device=device, finetune_mode=args.finetune_mode)
        else:
            model, preprocess = clip.load(args.model_name, device=device)


        model.eval()

        model_name = format_model_name(cp_path)
        method = args.eval_method

        if method == 'cosine':
            avg_pos, avg_neg, margin = evaluate_cosine_similarities(model, test_loader, device)
            print(f"Model: {model_name} | Eval: cosine")
            print(f"  Avg_Pos: {avg_pos:.4f}, Avg_Neg: {avg_neg:.4f}, Margin: {margin:.4f}")
            results.append({
                'Model': model_name,
                'Eval_Method': 'cosine',
                'Avg_Pos': avg_pos,
                'Avg_Neg': avg_neg,
                'Margin': margin
            })

        elif method == 'plot':
            avg_pos, avg_neg, margin, _ = evaluate_cosine_similarities_and_plot(
                model,
                test_loader,
                device,
                plot_title=f"Cosine Similarity Matrix: {model_name}",
                figsize=(12, 6),
                save_plot=True,
                plot_path=f"combined_similarity_matrix_{model_name}.png"
            )
            print(f"Model: {model_name} | Eval: plot")
            print(f"  Avg_Pos: {avg_pos:.4f}, Avg_Neg: {avg_neg:.4f}, Margin: {margin:.4f}")
            results.append({
                'Model': model_name,
                'Eval_Method': 'plot',
                'Avg_Pos': avg_pos,
                'Avg_Neg': avg_neg,
                'Margin': margin
            })

        elif method == 'random':
            avg_pos, avg_neg, avg_rand_neg, margin = evaluate_cosine_similarities_random_negative(
                model, test_loader, device
            )
            print(f"Model: {model_name} | Eval: random")
            print(f"  Avg_Pos: {avg_pos:.4f}, Avg_Neg: {avg_neg:.4f}, Avg_Rand_Neg: {avg_rand_neg:.4f}, Margin: {margin:.4f}")
            results.append({
                'Model': model_name,
                'Eval_Method': 'random',
                'Avg_Pos': avg_pos,
                'Avg_Neg': avg_neg,
                'Margin': margin,
                'Avg_Rand_Neg': avg_rand_neg
            })

        elif method == 'thresholds':
            accs = evaluate_thresholds_accuracy(
                model, test_loader, device, thresholds
            )
            print(f"Model: {model_name} | Eval: thresholds")
            row = {'Model': model_name}
            for th, a in accs.items():
                print(f"  Acc@{th}: {a:.4f}")
                row[f"Acc@{th}"] = round(a, 4)
            results.append(row)

        elif method == 'random+thresholds':
            avg_pos, avg_neg, avg_rand, margin, accs = evaluate_random_and_thresholds(
                model, test_loader, device, thresholds
            )
            # print(f"Model: {model_name} | Eval: random+thresholds")
            # print(f"  Avg_Pos: {avg_pos:.4f}, Avg_Neg: {avg_neg:.4f}, "
            #       f"Avg_Rand_Neg: {avg_rand:.4f}, Margin: {margin:.4f}")
            row = {
                'Model': model_name,
                'Avg_Pos': round(avg_pos, 4),
                'Avg_Neg': round(avg_neg, 4),
                'Avg_Rand_Neg': round(avg_rand, 4),
                'Margin': round(margin, 4)
            }
            for th, a in accs.items():
                print(f"  Acc@{th}: {a:.4f}")
                row[f"Acc@{th}"] = round(a, 4)
            results.append(row)

        else:
            print(f"Unknown evaluation method: {method}")


    df = pd.DataFrame(results).drop(columns=['Eval_Method'], errors='ignore')
    df = df.round(4)
    cols = ['Model', 'Avg_Pos', 'Avg_Neg', 'Margin', 'Avg_Rand_Neg'] + \
           [f"Acc@{th}" for th in thresholds]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print(df.to_string(index=False))
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved results to '{args.output_csv}'")

if __name__ == "__main__":
    main()
