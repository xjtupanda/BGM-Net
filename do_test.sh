collection=charades
visual_feature=i3d_rgb_lgi
root_path=./data/netdisk
model_dir=charades-debug-2023_03_11_18_51_05
#model_dir=charades-debug-2023_03_08_10_06_46



python method/eval.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir \