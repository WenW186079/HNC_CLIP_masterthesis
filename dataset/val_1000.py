import json
import random

with open("./HNC/hnc_val_sampled_1_percent.json", 'r') as f:
    data = json.load(f)

all_samples = []
for image_id, image_data in data.items():
    captions = image_data.get("captions", {})
    for caption_id, caption_data in captions.items():
        sample = {
            "image_id": image_id,
            "caption_id": caption_id,
            **caption_data
        }
        all_samples.append(sample)

print(f"Total samples available: {len(all_samples)}")

random.seed(42)

if len(all_samples) < 1000:
    print("Warning: Fewer than 1000 samples are available. Using all samples for validation.")
    val_samples = all_samples
    test_samples = []
else:
    # Sample 1000 samples for validation.
    val_samples = random.sample(all_samples, 1000)

    val_ids = {(sample["image_id"], sample["caption_id"]) for sample in val_samples}

    test_samples = [sample for sample in all_samples if (sample["image_id"], sample["caption_id"]) not in val_ids]

with open('val_dataset_1000.json', 'w') as f:
    json.dump(val_samples, f, indent=2)

with open('test_dataset.json', 'w') as f:
    json.dump(test_samples, f, indent=2)

print("Validation dataset with 1000 samples has been saved to 'val_dataset.json'.")
print(f"Test dataset with {len(test_samples)} samples has been saved to 'test_dataset.json'.")


#######
# Total samples available: 18962
# Validation dataset with 1000 samples has been saved to 'val_dataset.json'.
# Test dataset with 17962 samples has been saved to 'test_dataset.json'.
