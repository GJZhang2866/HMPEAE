import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str,
                        help="pre-trained language model")
    parser.add_argument("--dataset_type", default="rams", choices=["ace_eeqa", "rams", "wikievent", 'MLEE'], type=str,
                        help="dataset type. Both sentence-level(ace_eeqa) and document-level(rams/wikievent)")
    parser.add_argument("--role_path", default='./data/dset_meta/description_rams.csv', type=str,
                        help="a file containing all role names. Read it to access all argument roles of this dataset")
    parser.add_argument("--prompt_path", default='./data/prompts/prompts_rams_full.csv', type=str,
                        help="a file containing all prompts we use for this dataset")
    parser.add_argument("--output_dir", default='./outputs_res', type=str,
                        help="output folder storing checkpoint and all sorts of log files")
    parser.add_argument("--keep_ratio", default=1.0, type=float,
                        help="The ratio of remaining traning samples. We drop the others. Used in Few-shot setting.")
    parser.add_argument('--inference_only', default=False, action="store_true",
                        help="The model will inference directly without training if it were set as True")
    parser.add_argument('--single', default=False, action="store_true",
                        help="The model will extract one event at a time if set as True")

    parser.add_argument("--pad_mask_token", default=0, type=int,
                        help="padding token id")
    parser.add_argument('--logging_steps', default=100, type=int,
                        help="step intervals for outputting log files")
    parser.add_argument('--eval_steps', default=500, type=int,
                        help="step intervals for validation")
    parser.add_argument("--max_span_length", default=10, type=int,
                        help="a heuristic constraint: the maximum length of extracted arguments")
    parser.add_argument("--batch_size", default=4, type=int, 
                        help="batch size during training. with BP")
    parser.add_argument("--infer_batch_size", default=32, type=int, 
                        help="batch size during inference. without BP")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_enc_seq_length", default=500, type=int,
                        help="maximum length for context")
    parser.add_argument("--window_size", default=250, type=int,
                        help="for document exceeding the length constraint, add a window centering at the trigger word and drop words outside this window")
    parser.add_argument('--num_prompt_pos', default=0, type=int)
    parser.add_argument('--num_event_embed', default=0, type=int)
    parser.add_argument('--context_representation', default="decoder", choices=['encoder', 'decoder'], type=str,
                        help="whether use the full BART (decoder) or only BART-encoder (encoder) to represent the context.")

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--max_steps", default=20000, type=int)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--device", default='cuda', type=str)


    # setting only for the situation when inference_only
    parser.add_argument('--inference_model_path', default="/root/siton-data-zhangguangjunData/zgj/TabEAE-main/exp/div/主实验/rams/rams-large_2_sem_iter_800/checkpoint", type=str,
                        help="The path of checkpoint used for inference.")
    # setting only for base model.
    parser.add_argument("--max_dec_seq_length", default=360, type=int,
                        help="maximum length for single prompt")
    parser.add_argument("--max_span_num", default=1, type=int,
                        help="maximum arguments extracted for one role.")
    parser.add_argument('--th_delta', default=.0, type=float,
                        help="threshold controlling whether accept a candiate span as argument or not")
    # setting only for paie model
    parser.add_argument("--max_prompt_seq_length", default=64, type=int,
                        help="maximum length for multi-prompt")
    parser.add_argument('--matching_method_train', default="max", choices=["max", 'accurate'], type=str,
                        help="start/end token matching method during training.")
    parser.add_argument('--bipartite', default=True, action="store_true",
                        help="whether use bipartite matching loss during training or not.")
    parser.add_argument('--bert', default=False, action="store_true")
    
    parser.add_argument("--num_proto_per_type", default=2, type=int)
    parser.add_argument("--hpnfile", default='prototypes_rams/large/prototypes-1024d-66c_mutil2_proto_sem.npy', type=str)
    parser.add_argument("--role2id_file", default='data/dset_meta/role2id_rams.json', type=str)
    parser.add_argument("--pos_loss_weight", default=1, type=int)
    parser.add_argument("--max_iter", default=800, type=int)
    args = parser.parse_args()

    # if args.inference_only:
    #     args.output_dir = "/".join(args.inference_model_path.split("/")[:-1])
    return args