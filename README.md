# Everchanging Language Models Dev Repo


In order to reproduce our results, take the following steps:
### 1. Create conda environment and install requirements
```
conda create -n elm python=3.8 && conda activate elm
pip install -r requirements.txt
```

Also, make sure to install the correct version of pytorch corresponding to the CUDA version and environment:
Refer to https://pytorch.org/
```
#For CUDA 10.x
pip3 install torch torchvision torchaudio
#For CUDA 11.x
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Download the data used for the experiments.
To download Full Wikipedia data for 2021.08-2021.12:
```
wget https://continual.blob.core.windows.net/elm/Wikipedia_Full.zip
```

To download TWiki_Diffsets data for 2021.08-2021.12:
```
wget https://continual.blob.core.windows.net/elm/TWiki_Diffsets.zip
```

To download TWiki_Probes data for 2021.08-2021.12:
```
wget https://continual.blob.core.windows.net/elm/TWiki_Probes.zip
```

Download the data to ```data``` and unzip it

This is an example of performing continual pretraining on **TWiki_Diffsets** (main experiment) with **CKL**
```
python run.py --config configs/baseline_gpt2_s.json
```
This is an example of performing light-tuning pretrained model
```
python run.py --config configs/baseline_gpt2_s.json
```
This is an example of getting the **TWiki_Probes New** zero-shot evaluation of continually pretrained **CKL**
```
python run.py --config configs/evaluation/GPT2/subset/0801-0901_new.json
```

## Generation of Datasets

For Generation of Wikipedia_Full, TWiki_Diffsets, TWiki_Probes, please refer to the link below.
```
https://github.com/CHLee0801/TemporalWikiDatasets
```