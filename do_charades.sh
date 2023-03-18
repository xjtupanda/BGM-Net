collection=charades
visual_feature=i3d_rgb_lgi
clip_scale_w=0.5
frame_scale_w=0.5
exp_id=debug
root_path=./data/netdisk
device_ids=5
use_matcher_start_epoch=75
map_size=32
# training

python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids $device_ids \
                    --use_matcher_start_epoch $use_matcher_start_epoch --map_size $map_size