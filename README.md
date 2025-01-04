# HNC_CLIP_masterthesis

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

## Download HNC dataset
```
git clone https://huggingface.co/datasets/patilli/HNC
```

## Download GQA dataset
```
mkdir gqa_dataset
cd gqa_dataset

wget https://nlp.stanford.edu/data/gqa/images.zip
unzip images.zip -d ./images
```
