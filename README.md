# HNC_CLIP_masterthesis

## Set an environment
```
python -m venv the_env
source /mount/arbeitsdaten/deepfake/SpeechTechnology2023/mm/the_env/bin/activate
```

## Install the packages
- To install the required dependencies in your environment run: 
```
bash install.sh
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
