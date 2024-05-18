collection=charades
visual_feature=i3d_rgb_lgi
root_path=./data/netdisk
model_dir=charades-debug-2023_12_15_21_30_42
device_ids=7


python method/eval.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir \