collection=tvr
visual_feature=i3d_resnet
q_feat_size=768
margin=0.1
bsz=128
lr=2.5e-4
map_size=32
smp_rate=0.01
exp_id=tune
root_path=./data/netdisk
device_ids=0
use_matcher_start_epoch=0
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --q_feat_size $q_feat_size --margin $margin --device_ids $device_ids \
                    --bsz $bsz --lr $lr --use_matcher_start_epoch $use_matcher_start_epoch --map_size $map_size --smp_rate $smp_rate