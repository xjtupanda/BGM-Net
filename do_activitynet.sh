collection=activitynet
visual_feature=i3d
exp_id=redebug
root_path=./data/netdisk
device_ids=0
use_matcher_start_epoch=20
map_size=48
bsz=128
smp_rate=1.0
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --device_ids $device_ids \
                    --use_matcher_start_epoch $use_matcher_start_epoch --map_size $map_size --smp_rate $smp_rate --bsz $bsz