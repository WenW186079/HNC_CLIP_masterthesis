import json

json_file_path = './HNC/hnc_clean_strict_val.json'

def search_hnc_negatives(json_file_path, target_image_id, target_caption):

    with open(json_file_path, 'r') as f:
        annotations = json.load(f)  
        if target_image_id not in annotations:
            print(f"Image ID {target_image_id} not found.")
            return

        image_data = annotations[target_image_id]["captions"]

        hnc_negatives = []
        for cap_id, cap_data in image_data.items():
            caption = cap_data["caption"]
            label = cap_data["label"]

            if caption == target_caption and label == 1:
                print(f"Positive Caption Found: {caption}")
            elif label == 0:
                hnc_negatives.append(caption)

        if hnc_negatives:
            print(f"\nHNC Negatives for '{target_caption}':")
            for neg_caption in hnc_negatives:
                print(f"  - {neg_caption}")
        else:
            print(f"No HNC negatives found for '{target_caption}'.")

# Example
image_id = "2415537" 
positive_caption = "The trees are to the left of the flag."

search_hnc_negatives(json_file_path, image_id, positive_caption)
