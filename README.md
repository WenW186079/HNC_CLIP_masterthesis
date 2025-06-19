# HNC_CLIP_masterthesis
The primary goal of this research is to enhance the relational understanding capabilities of CLIP while preserving its strong object recognition performance. This will be achieved by fine-tuning CLIP using the Hard Negative Captions (HNC) dataset, which provides positive samples paired with minimally contradictory negative samples. The fine-tuning process will involve training the model to minimize the similarity between an image and its corresponding positive caption while maximizing the similarity between the image and its hard negative caption and other random negative captions. Based on the traditional CLIP loss function, the hard negative loss will be added with weight. By using negative samples that closely resemble the positive ones, the model will be forced to capture finer-grained features and develop a more nuanced understanding of object relationships. The key objective is to create a fine-tuned version of CLIP that excels in both object recognition and relational reasoning, overcoming the limitations of the original model. The resulting model will be a robust vision-language system capable of handling complex scenarios requiring relational awareness, with potential applications in tasks like Visual Question Answering (VQA) and other multimodal reasoning tasks. 

Details seen in [slides](reports/slides.pdf)

# Loss Fucntions
This repository implements six alternative loss functions for CLIP-style training:

- **StandardCLIPLoss**  
  
The vanilla CLIP objective: it normalizes image and text embeddings, computes their cosine similarities (scaled by a learnable logit_scale), and applies symmetric cross‐entropy (image→text and text→image) to align matching pairs and repel all other pairs.

- **CLIPLossL2**  
  
Extends the standard CLIP loss by explicitly incorporating hard negative capture and an optional L₂ parameter‐regularization term. It also supports a dynamic schedule for increasing the hard‐negative weight over training steps.
  
- **CLIPLossKL**  
  
Similar to CLIPLossL2’s hard‐negative treatment, but replaces the L₂ regularizer with a KL‐divergence distillation term: a frozen “teacher” CLIP model provides soft (temperature‐scaled) similarity distributions over positives and negatives, which the student minimizes via KL divergence alongside the standard contrastive losses.

- **DPOCLIPLoss**  
  
Implements a direct preference optimization (DPO) loss over (image, positive, negative) triplets: it measures how much the model’s positive–negative score gap improves over the teacher model(clip), and applies a logistic (binary cross‐entropy) penalty to encourage higher preference margins, without any contrastive term.

- **DPOContrastiveCLIPLoss**  
  
Combines the DPO logistic loss from DPOCLIPLoss with the standard CLIP contrastive loss: a weighted sum of the two (controlled by alpha) ensures both strong pairwise alignment and improved preference margins relative to a reference model.

- **CombinedCLIPDPOLoss**  
  
It computes the standard CLIP contrastive loss and a DPO logistic loss on (image, positive, negative) gaps, plus an optional L₂ regularization on model parameters. Final loss is an alpha-weighted mixture of contrastive and DPO+regularization objectives.

# Select finetuning parameters
Control which parts of CLIP are trainable:

| Mode                   | Token Embedding | Text Encoder    | Text Projection | Vision Encoder   | Vision Projection |
|------------------------|-----------------|-----------------|-----------------|------------------|-------------------|
| **text_encoder**       | True            | True            | True            | –                | -                 |
| **vision_encoder**     | –               | –               | –               | True             | True              |
| **full_encoder**       | True            | True            | True            | True             | True              |
| **last_encoder**       | –               | True (last block)| –              | True (last block)| –                 |


# Experiments
## Setup
### Set an environment
```
python3.10 -m venv thesis_env
source /mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/data/thesis_env/bin/activate 
```

### Install the packages
- To install the required dependencies in your environment run: 
```
pip install -r requirements.txt
```

### Download HNC dataset
Thanks to [HNC group(Esra Dönmez, Pascal Tilli, Hsiu-Yu Yang, Ngoc Thang Vu and Carina Silberer)](https://github.com/DigitalPhonetics/hard-negative-captions)
```
mkdir data &&
cd data &&
git clone https://huggingface.co/datasets/patilli/HNC
```

Here I used split of this dataset
```
git clone https://huggingface.co/datasets/WenWW/HNC_sample
```

### Download GQA dataset
```
mkdir gqa_dataset &&
cd gqa_dataset &&
wget https://nlp.stanford.edu/data/gqa/images.zip &&
unzip images.zip -d ./images
```

### Download Coco dataset for evaluation
```
mkdir Coco &&
cd Coco &&
wget http://images.cocodataset.org/zips/val2014.zip &&
unzip val2014.zip -d ./val2014 
```
And it's test [json file](dataset/test_coco_aug_withneg.json). Thanks to [Structure-CLIP](https://github.com/zjukg/Structure-CLIP?tab=readme-ov-file) group

### Download CLIP model using git
```
pip install git+https://github.com/openai/CLIP.git
```

### Overview of the project directory and its main components
```
thesis/          
├── config/                     
├── data/                      
│   ├── Coco/                  
│   ├── gqa_dataset/            
│   └── HNC/                   
├── eval/                      
│   ├── eval_functions.py       
│   ├── evaluation.py           
│   └── run_test_data.sh        
├── models/                    
├── load_data.py                
├── loss_func.py                
├── mainCLIP.py                
└── trainCLIP.py               
```


## Train CLIP model with hard negative captions
### Login wandb
```
wandb login 
```

### Use deepspeed run the code
```
# Adjust the relevent path
CUDA_VISIBLE_DEVICES=0,1 deepspeed mainCLIP.py --config_path config/config.yaml
```

## Evaluation
Test datasets 
- GQA test dataset
- Coco test dataset

Approach 1: Intrinsic metric
- Average Positive Cosine Similarity
- Average Negative Cosine Similarity
- Margin (Positive - Negative)
- Average Random Negative Similarity

Approach 2: Distinguishing posivite text from negative text accuracy

Distinguishing accuracy is measured by comparing how much more similar an image is to its true caption than to a hard negative. For each sample, we compute
ratio = (cosine_similarity(image, positive_caption)) / (cosine_similarity(image, negative_caption) + ε) and count it as correct if ratio ≥ threshold.

```
chmod +x run_test_data.sh

# Adjust the file path, mode...and run the script
./run_test_data.sh
```

# Result and analysis
Details can be found [here](result_and_analysis)

# Models
All models can be found [here](https://drive.google.com/drive/folders/11Pxr9IA4l4EegGzgcvBzVNPWCxFmqsGl?usp=drive_link) with relative [config](config) files

# Explanation
Here I used second-order attribution pipeline, thanks to [Pascal Tilli, Lucas Moelleret al.](https://arxiv.org/abs/2408.14153)

