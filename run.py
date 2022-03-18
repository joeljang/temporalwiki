import argparse
from argparse import ArgumentParser
import os
import json
import random
from evaluation import evaluate
from evaluation_ppl import evaluate_ppl
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import T5Tokenizer, GPT2Tokenizer
from models import load_model

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Setting GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    #Init configs that are not given
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.5
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.01
    if 'output_log' not in hparam:
        hparam.output_log = None
    if 'len_data' not in hparam:
        hparam.len_data = None
    if 'num_files' not in hparam:
        hparam.num_files = 1
    if 'learning_rate' not in hparam:
        hparam.learning_rate = None
    if 'gradient_accumulation_steps' not in hparam:
        hparam.gradient_accumulation_steps = 0
    if 'num_train_epochs' not in hparam:
        hparam.num_train_epochs = 0
    if 'use_lr_scheduling' not in hparam:
        hparam.use_lr_scheduling = False
    if 'num_workers' not in hparam:
        hparam.num_workers = 0
    if 'output_dir' not in hparam:
        hparam.output_dir = None
    if 'wandb_log' not in hparam:
        hparam.wandb_log = False
    if 'accelerator' not in hparam:
        hparam.accelerator = None
    if 'checkpoint_path' not in hparam:
        hparam.checkpoint_path =''
    if 'resume_from_checkpoint' not in hparam:
        hparam.resume_from_checkpoint = None
    if 'fp16' not in hparam:
        hparam.fp16 = False

    #Logging into WANDB if needed
    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name, entity="lklab_kaist")
    else:
        wandb_logger = None
        
    #Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir, # Path to save the checkpoints
        dataset=hparam.dataset,
        dataset_version = hparam.dataset_version,
        len_data = hparam.len_data,
        model_name_or_path=hparam.model,
        method=hparam.method,
        mode=hparam.mode,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        learning_rate=hparam.learning_rate,
        weight_decay=hparam.weight_decay,
        adam_epsilon=1e-8,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.train_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        num_files=hparam.num_files,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint, 
        use_lr_scheduling = hparam.use_lr_scheduling,
        val_check_interval = 1.0,
        fp16=hparam.fp16,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=hparam.grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        check_validation_only=hparam.check_validation,
        checkpoint_path=hparam.checkpoint_path,
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
    )
    args = argparse.Namespace(**args_dict)

    #Setting different val & checkpoint saving config for mode
    if args.mode=='pretrain_brute':
        saving_epoch = args.num_files
    else:
        saving_epoch = 1

    # Defining how to save model checkpoints during training. Details: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html 
    callbacks = [ModelCheckpoint(dirpath = args.output_dir, every_n_epochs=saving_epoch,save_top_k=-1)]
    checkpoint_callback = True

    if args.output_dir=="":
        checkpoint_callback = False # Do not save model checkpoints when output dir is empty
        callbacks=[]

    # Logging Learning Rate Scheduling
    if args.use_lr_scheduling and hparam.wandb_log:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    

    # Setting Flags for pytorch lightning trainer. Details: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=int(args.num_train_epochs * args.num_files),
        precision= 16 if args.fp16 else 32,
        amp_backend="native",
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        enable_checkpointing=checkpoint_callback,
        check_val_every_n_epoch= saving_epoch,
        logger = wandb_logger,
        callbacks = callbacks,
        strategy=args.accelerator
    )
    if 't5' in args.model_name_or_path:
        Model = load_model('T5')
    elif 'gpt2' in args.model_name_or_path: 
        Model = load_model('GPT2')
    else:
        raise Exception('currently not supporting given model')
    
    if args.check_validation_only:
        if 'evaluate_ppl' in args.mode:
            evaluate_ppl(args, Model)
        elif args.mode == 'evaluate':
            evaluate(args, Model)
    else:
        set_seed(40)
        if args.checkpoint_path!="":
            model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False) 
        else:
            model = Model(args)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
