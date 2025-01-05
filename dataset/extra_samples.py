import json

def display_top_n_samples(json_file_path, n=5):
    try:
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)  
            keys = list(data.keys())  
            print(f"\nTotal entries in JSON: {len(keys)}")
            print(f"\nShowing the first {n} samples:\n")
            
            
            for i in range(min(n, len(keys))):
                image_name = keys[i]
                print(f"Sample {i+1}: Image File: {image_name}")
                print(json.dumps(data[image_name], indent=4))  
                print("\n" + "-"*80 + "\n")
    
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


json_file_path = "/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_clean_strict_train.json"
display_top_n_samples(json_file_path, n=5)
