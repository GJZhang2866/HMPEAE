from transformers import BertTokenizer, BertModel, RobertaModel,  RobertaTokenizer
import torch
import json
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

meta_file = 'data/dset_meta/role_num_rams.json'
output_role2id_path ='./role2id.json'
output_semnatic_vectors_path ='./semnatic_vectors.npy'

with open(meta_file) as f:
    meta = json.load(f)
all_role_list = ['None role']
for event_type, roles in meta.items():
    role_list = list(roles.keys())
    for role in role_list:
        if role not in all_role_list:
            all_role_list.append(role)
role2id = {}          
for i,role in enumerate(all_role_list):
    role2id[role] = i
with open(output_role2id_path,'w',encoding="utf8")as f:
    json.dump(role2id,f)
role_embs = []
for role in all_role_list:
    role_tokens = tokenizer.tokenize(role)
    role_tokens = [tokenizer.cls_token] + role_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(role_tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.pooler_output
        role_embs.append(hidden_states.squeeze(0).cpu().numpy())
            
np.save(output_semnatic_vectors_path,  np.array(role_embs))
