if [ $# == 0 ] 
then
    SEED=41
    LR=2e-5
    BatchSize=4
else
    SEED=$1
    LR=$2
fi


nppt=2

work_path=exp/pred/rams-large_proto_${nppt}_800_div_target2
mkdir -p $work_path

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --dataset_type=rams \
    --context_representation=decoder \
    --model_name_or_path=roberta-large \
    --role_path=./data/dset_meta/description_rams.csv \
    --prompt_path=./data/prompts/prompts_rams_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --batch_size=$BatchSize \
    --max_steps=10000 \
    --max_enc_seq_length 500 \
    --max_dec_seq_length 200 \
    --window_size 260 \
    --warmup_steps 0.1 \
    --bipartite \
    --num_prompt_pos 10 \
    --hpnfile prototypes_rams/large/prototypes-1024d-66c_mutil${nppt}_proto.npy \
    --num_proto_per_type ${nppt} \
    --role2id_file data/dset_meta/role2id_rams.json \
    --max_iter 800 


