if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
    BatchSize=4
else
    SEED=$1
    LR=$2
fi

work_path=Infer/wikievent/
mkdir -p $work_path

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --dataset_type=wikievent \
    --context_representation=decoder \
    --model_name_or_path=roberta-large \
    --inference_model_path=wikievent-large/checkpoint \
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
    --inference_only \
    --single \
    --hpnfile prototypes_wiki/large/prototypes-1024d-81c_mutil3_proto.npy \
    --num_proto_per_type 3 \
    --role2id_file data/dset_meta/role2id_wikievent.json
