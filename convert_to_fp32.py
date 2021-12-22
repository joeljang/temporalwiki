from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import os
# lightning deepspeed has saved a directory instead of a file

#checkpoint_dir = 'outputs/T5_base_0809_lr1e-4'
#checkpoint_dir = 'outputs/T5_base_0809_lr3e-5'
checkpoint_dir = 'outputs/T5_base_0809_lr1e-5'
#output_path = 'outputs/T5_base_0809_1e-4/'
#output_path = 'outputs/T5_base_0809_3e-5/'
output_path = 'outputs/T5_base_0809_1e-5/'

if not os.path.isdir(output_path):
    os.mkdir(output_path)

lst = os.listdir(checkpoint_dir)

for l in lst:
    file = checkpoint_dir+'/'+l
    out = output_path + (l.split('-'))[0]
    convert_zero_checkpoint_to_fp32_state_dict(file, out)
