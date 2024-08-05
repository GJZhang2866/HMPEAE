if [ $# == 0 ] 
then
    SEED=43
    LR=2e-5
    BatchSize=4
else
    SEED=$1
    LR=$2
fi

num_proto_per_type=3
work_path=exp/div/wikievent-large_prot${num_proto_per_type}
mkdir -p $work_path

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --dataset_type=wikievent \
    --context_representation=decoder \
    --model_name_or_path=roberta-large \
    --role_path=./data/dset_meta/description_wikievent.csv \
    --prompt_path=./data/prompts/prompts_wikievent_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --batch_size=$BatchSize \
    --max_steps=10000 \
    --max_enc_seq_length 500 \
    --max_dec_seq_length 360 \
    --window_size 250 \
    --bipartite \
    --hpnfile prototypes_wiki/large/prototypes-1024d-81c_mutil${num_proto_per_type} _proto_sem.npy \
    --num_proto_per_type ${num_proto_per_type} \
    --role2id_file data/dset_meta/role2id_wikievent.json \
    --max_iter 100

