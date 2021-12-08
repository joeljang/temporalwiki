from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# lightning deepspeed has saved a directory instead of a file
save_path = "/home/joel/everchange-dev/outputs/T5_base_08_lr3e-5/epoch=0-step=18856.ckpt"
output_path = "/home/joel/everchange-dev/outputs/T5_base_08_lr3e-5/epoch0.ckpt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)