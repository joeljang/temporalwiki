from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import os
# lightning deepspeed has saved a directory instead of a file

checkpoint_dir = 'outputs/GPT2_large_10_3e-5_ckl'
output_path = 'outputs/GPT2_large_10_3e-5_ckl_/'

if not os.path.isdir(output_path):
    os.mkdir(output_path)

lst = os.listdir(checkpoint_dir)

for l in lst:
    file = checkpoint_dir+'/'+l
    out = output_path + (l.split('-'))[0]
    convert_zero_checkpoint_to_fp32_state_dict(file, out)
