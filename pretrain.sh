export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CONFIG='./cfgs/model_cfg'
#choose the model cfg in ./cfgs
OUTPUT='./ckpt/save_ckpt'

DATA='/path_to_your_data'

python  -m torch.distributed.launch\
        --nproc_per_node 8\
        main.py\
        --cfg $CONFIG\
        --data-path $DATA\
        --output $OUTPUT
#if use PVT add "--find-unused-params"

