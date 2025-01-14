export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CONFIG='./cfgs/pola_swin_t.yaml'

OUTPUT='./ckpt/pola_swin_t/'

DATA='/path/to/data/'

python  -m torch.distributed.launch\
        --nproc_per_node 8\
        --master_port 7194\
        main.py\
        --cfg $CONFIG\
        --data-path $DATA\
        --output $OUTPUT
