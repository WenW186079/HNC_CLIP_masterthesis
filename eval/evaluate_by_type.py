import os
import torch
import clip
import pandas as pd
import sys

from eval_functions import evaluate_by_type_random_thresholds_total

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from load_data import TestTypeDataset, TypeTestDatasetWrapper

#  =========================================
CHECKPOINT_PATH = "/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/data/thes/models/C_DPO_KL_S_full/epoch_1_full_encoder.pt"  # or "base"
MODEL_NAME = "ViT-B/32"
TEST_JSON = "/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/data/thes/data/HNC/hnc_clean_strict_test.json"
IMAGES_PATH = "/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/data/thes/data/gqa_dataset/images/images"
OUTPUT_CSV = "./eval_by_type_result.csv"
BATCH_SIZE = 32
THRESHOLDS = [1, 1.1, 1.2, 1.5, 2, 3]
# =========================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)

    if CHECKPOINT_PATH != "base" and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded model checkpoint from {CHECKPOINT_PATH}")
    else:
        print(" Using base CLIP model (no fine-tuning checkpoint loaded)")

    test_dataset = TestTypeDataset(
        json_path=TEST_JSON,
        image_folder=IMAGES_PATH,
        tokenizer=clip.tokenize,
        transform=preprocess
    )
    type_wrapper = TypeTestDatasetWrapper(test_dataset)

    df = evaluate_by_type_random_thresholds_total(
        model=model,
        type_wrapper=type_wrapper,
        device=device,
        thresholds=THRESHOLDS,
        batch_size=BATCH_SIZE
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved results to {OUTPUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
