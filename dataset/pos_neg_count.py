import json

json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_clean_strict_train.json'

with open(json_file_path, 'r') as f:
    annotations = json.load(f)

positive_count = 0
negative_count = 0

for img_id, data in annotations.items():
    captions_dict = data.get("captions", {})
    
    for cap_id, cap_data in captions_dict.items():
        if cap_data["label"] == 1:  
            positive_count += 1
        elif cap_data["label"] == 0:  
            negative_count += 1

print(f"Total Positive Samples: {positive_count}")
print(f"Total Negative Samples: {negative_count}")

'''
=== Result ===

train:
Total Positive Samples: 8208196
Total Negative Samples: 8208196

val:
Total Positive Samples: 1157416
Total Negative Samples: 1157416
'''
