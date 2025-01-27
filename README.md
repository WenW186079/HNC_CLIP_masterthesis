# HNC_CLIP_masterthesis
The primary goal of this research is to enhance the relational understanding capabilities of CLIP while preserving its strong object recognition performance. This will be achieved by fine-tuning CLIP using the Hard Negative Captions (HNC) dataset, which provides positive samples paired with minimally contradictory negative samples. The fine-tuning process will involve training the model to minimize the similarity between an image and its corresponding positive caption while maximizing the similarity between the image and its hard negative caption and other random negative captions. Based on the traditional CLIP loss function, the hard negative loss will be added with weight. By using negative samples that closely resemble the positive ones, the model will be forced to capture finer-grained features and develop a more nuanced understanding of object relationships. The key objective is to create a fine-tuned version of CLIP that excels in both object recognition and relational reasoning, overcoming the limitations of the original model. The resulting model will be a robust vision-language system capable of handling complex scenarios requiring relational awareness, with potential applications in tasks like Visual Question Answering (VQA) and other multimodal reasoning tasks. 

Details seen in [proposal](Proposal.pdf)

# Experiments

## Set an environment
```
python3.10 -m venv negcl_env
source /mount/studenten/team-lab-cl/data2024/w/data/negcl_env/bin/activate
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
unzip val2014.zip -d ./val2014

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

### Download CLIP model using git
```
pip install git+https://github.com/openai/CLIP.git
```

## Use HNC tune CLIP model
### Set cache path
```
export TORCH_EXTENSIONS_DIR=/mount/studenten/team-lab-cl/data2024/w/data/torch_extensions/
echo $TORCH_EXTENSIONS_DIR

```
### Login Huggingface and wandb
```
huggingface-cli login
wandb login 
```

### Use deepspeed run the code

- ⚠️[TODO]: USE ARG
- ⚠️[TODO]: find best hyper parameters
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main_deepspeed.py
```

## Evaluation
Approach 1: Distinguishing posivite text from negative text
- GQA dataset
- Coco dataset

```
chmod +x run_evaluation.sh

# Adjust the file path and run the script
./run_evaluation.sh
```

Approach 2: Retrieval 
- GQA dataset
- Coco dataset
```
#⚠️TODO
```

Approach 3: Downstream task
```
#⚠️TODO
```
