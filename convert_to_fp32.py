from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# lightning deepspeed has saved a directory instead of a file
save_path = “outputs/T5_base_ssm08lr”
output_path = “lightning_logs/version_0/checkpoints/epoch=0-step=0_2.ckpt”
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)