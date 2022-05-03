# TemporaWiki
Official code for the paper [TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models](https://arxiv.org/abs/2204.14211)

In order to generate new TemporalWiki (training and evaluation corpus), use the [TemporalWikiDatasets](https://github.com/CHLee0801/TemporalWikiDatasets) repository.

In order to reproduce our results, take the following steps:
### 1. Create conda environment and install requirements
```
conda create -n twiki python=3.8 && conda activate twiki
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

### 2. Download the preprocessed training and evaluation data (5 snapshots from 2021.08 - 2021.12) used for the experiments on the paper.
To download the Entire Wikipedia Corpus data:
```
wget https://continual.blob.core.windows.net/elm/Wikipedia_Full.zip
```

To download TWiki_Diffsets:
```
wget https://continual.blob.core.windows.net/elm/TWiki_Diffsets.zip
```

To download TWiki_Probes:
```
wget https://continual.blob.core.windows.net/elm/TWiki_Probes.zip
```

Download the data to ```data``` directory and unzip it

Finally, download the Initial GPT-2 model checkpoint trained on 08.2021 Wikipedia Snapshot used as the initial model for the paper.
```
wget https://continual.blob.core.windows.net/elm/model_checkpoints/08/GPT2_large_08_full.ckpt
```


### 3. Run the experiment and configuration components
This is an example of performing continual pretraining on **TWiki_Diffsets** (main experiment) with **CKL**
```
python run.py --config configs/baseline_gpt2_s.json
```

After training the model, run ```convert_to_fp32.py``` to convert the fp16 model checkpoints to fp32 checkpoint files to be used for evaluation.

This is an example of performing light-tuning on the pretrained models
```
python run.py --config configs/light_tuning/GPT2/subset/0801-0901_new.json
```
This is an example of getting the **TWiki_Probes New** zero-shot evaluation of continually pretrained **CKL**
```
python run.py --config configs/evaluation/GPT2/subset/0801-0901_new.json
```

For components in configuration file, please refer to the [Continual-Knowledge-Learning](https://github.com/joeljang/continual-knowledge-learning) repository.

## Generation of Datasets

For Generation of Wikipedia_Full, TWiki_Diffsets, TWiki_Probes, please refer to the [TemporalWikiDatasets](https://github.com/CHLee0801/TemporalWikiDatasets)
