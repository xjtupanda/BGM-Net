collection=charades
visual_feature=i3d_rgb_lgi
clip_scale_w=0.6
frame_scale_w=0.4
exp_id=revision
root_path=./data/netdisk
device_ids=0
bsz=16
use_matcher_start_epoch=5
smp_rate=1.0
map_size=48
# training

python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids $device_ids \
                    --use_matcher_start_epoch $use_matcher_start_epoch --map_size $map_size --smp_rate $smp_rate --bsz $bsz