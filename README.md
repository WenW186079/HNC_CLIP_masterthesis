# HNC_CLIP_masterthesis
The primary goal of this research is to enhance the relational understanding capabilities of CLIP while preserving its strong object recognition performance. This will be achieved by fine-tuning CLIP using the Hard Negative Captions (HNC) dataset, which provides positive samples paired with minimally contradictory negative samples. The fine-tuning process will involve training the model to minimize the similarity between an image and its corresponding positive caption while maximizing the similarity between the image and its hard negative caption and other random negative captions. Based on the traditional CLIP loss function, the hard negative loss will be added with weight. By using negative samples that closely resemble the positive ones, the model will be forced to capture finer-grained features and develop a more nuanced understanding of object relationships. The key objective is to create a fine-tuned version of CLIP that excels in both object recognition and relational reasoning, overcoming the limitations of the original model. The resulting model will be a robust vision-language system capable of handling complex scenarios requiring relational awareness, with potential applications in tasks like Visual Question Answering (VQA) and other multimodal reasoning tasks. 

Details seen in [proposal](Proposal.pdf)

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

| Mode                 | Trainable                                                  | Frozen                                                           |
| -------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| **text_encoder**     | Text transformer<br>`text_projection` & `token_embedding`   | Visual encoder & `vision_projection`                             |
| **vision_encoder**   | Visual encoder & `vision_projection`                        | Text transformer,<br>`text_projection` & `token_embedding`       |
| **full_encoder**     | **All** parameters                                          | —                                                                |
| **full_encoder_last**| Last residual block of vision & text encoders<br>`vision_projection`<br>`text_projection` & `token_embedding` | Everything else                                                  |



# Experiments

## Set an environment
```
python3.10 -m venv thesis_env
source /mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/data/thesis_env/bin/activate 
```

## Install the packages
- To install the required dependencies in your environment run: 
```
pip install -r requirements.txt
```

### Download HNC dataset
```
git clone https://huggingface.co/datasets/patilli/HNC
```

### Download GQA dataset
```
mkdir gqa_dataset
cd gqa_dataset

wget https://nlp.stanford.edu/data/gqa/images.zip
unzip images.zip -d ./images
```

### Download Coco dataset for evaluation
```
mkdir Coco
cd Coco

wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
unzip val2014.zip -d ./val2014
unzip test2014.zip -d ./test2014

wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

### Download CLIP model using git
```
pip install git+https://github.com/openai/CLIP.git
```

## Use HNC tune CLIP model
### Login wandb
```
wandb login 
```

### Use deepspeed run the code
```
CUDA_VISIBLE_DEVICES=0,1 deepspeed mainCLIP.py --config_path config/config.yaml
```

## Evaluation
Approach 1: Intrinsic metric

Approach 2: Distinguishing posivite text from negative text
- GQA dataset
- Coco dataset

```
chmod +x run_test_data.sh

# Adjust the file path, mode...and run the script
./run_test_data.sh
```
