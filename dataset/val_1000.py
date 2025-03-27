import json
import random

with open("./HNC/hnc_val_sampled_1_percent.json", 'r') as f:
    data = json.load(f)

caption_pairs = []
for image_id, image_data in data.items():
    captions = image_data.get("captions", {})
    for caption_id in captions.keys():
        caption_pairs.append((image_id, caption_id))

total_samples = len(caption_pairs)
print(f"Total caption samples available: {total_samples}")

random.seed(42)

if total_samples < 1000:
    print("Warning: Fewer than 1000 samples available. Using all samples for validation.")
    val_pairs = set(caption_pairs)
else:
    val_pairs = set(random.sample(caption_pairs, 1000))

val_data = {}
test_data = {}

for image_id, image_data in data.items():
    captions = image_data.get("captions", {})
    val_captions = {}
    test_captions = {}
    for caption_id, caption_data in captions.items():
        if (image_id, caption_id) in val_pairs:
            val_captions[caption_id] = caption_data
        else:
            test_captions[caption_id] = caption_data
    if val_captions:
        val_data[image_id] = {"captions": val_captions}
    if test_captions:
        test_data[image_id] = {"captions": test_captions}

with open('val_dataset.json', 'w') as f:
    json.dump(val_data, f, indent=2)

with open('test_dataset00.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print("Validation dataset (1000 samples)")
print(f"Test dataset ({total_samples - len(val_pairs)} samples)")
#######
# Total samples available: 18962
# Validation dataset:1000 
# Test dataset: 17962 
