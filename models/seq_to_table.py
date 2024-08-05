# paie model
from itertools import chain
import torch
import torch.nn as nn
from .modeling_roberta_ import RobertaModel_, RobertaPreTrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple, seq_len_to_mask
from ot import sinkhorn
from torch.nn import CrossEntropyLoss
from sinkhorn import SinkhornDistance
import json
import torch.nn.functional as F
import einops

import torch.nn.functional as F
from torch.autograd import Variable
class Seq2Table(RobertaPreTrainedModel):
    def __init__(self, config, decode_layer_start=17, num_prompt_pos=0, num_event_embed=0,protos= None):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel_(config, decode_layer_start=decode_layer_start)
        self.decode_layer_start = decode_layer_start
        self.dec_input_drop = nn.Dropout(0.1)
        self.w_prompt_start = nn.Parameter(torch.zeros(config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.zeros(config.hidden_size, ))

        self.num_prompt_pos = num_prompt_pos
        if self.num_prompt_pos > 0:
            self.event_type_embed = nn.Embedding(num_prompt_pos, config.hidden_size, _weight=torch.zeros(num_prompt_pos, config.hidden_size), padding_idx=0)

        self.num_event_embed = num_event_embed
        if self.num_event_embed > 0:
            self.event_embed = nn.Embedding(num_event_embed, config.hidden_size, _weight=torch.zeros(num_event_embed, config.hidden_size), padding_idx=0)

        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')
        self.proto_ema_momentum = 0.9
        self.compact_weight = 0.1
        self.num_proto_per_type = config.num_proto_per_type
        self.role2id =config.role2id
        self.protos = nn.Parameter(protos.view(-1,self.config.hidden_size))

        pos_loss_weight = self.config.pos_loss_weight
        self.pos_loss_weight = torch.tensor([pos_loss_weight for _ in range(self.protos.size(0))])
        self.pos_loss_weight[0:self.num_proto_per_type] = 1
        self.max_iter = config.max_iter
    def reset(self):
        self.w_prompt_start = nn.Parameter(torch.rand(self.config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.rand(self.config.hidden_size, ))
        self.protos = nn.Parameter(self.protos)
       
        if self.num_prompt_pos > 0:
            self.roberta._init_weights(self.event_type_embed)

        if self.num_event_embed > 0:
            self.roberta._init_weights(self.event_embed)

    def sinkhorn_matching_and_ema(
            self,
            hiddens,
            logits,
            pred_tags,
            gts,
            role_ids
        ):
        def ema_update(v: float, new_v: float):
            momentum = self.proto_ema_momentum
            return v * momentum + new_v * (1 - momentum)

       
        well_pred_mask = pred_tags == gts
        logits = einops.rearrange(logits, "b (k p) -> b k p", p=self.num_proto_per_type)
        target_indexes = gts.clone()
        protos = self.protos.clone().to(logits.device)
        protos = einops.rearrange(protos, "(k p) d -> k p d", p=self.num_proto_per_type)
        for i, lable in enumerate(role_ids):
            mask_k = gts == lable # 得到 当前 lable 的mask，是当前lable的 片段为1 ，否则为0
            
            ty_logits = logits[mask_k, lable,:]      # 得到 预测为lable标签 span， 等于 lable 的logits  (预测为lable的span_num, 1 ,num_proto_per_type)
            cost = (1 - ty_logits)                  # 距离
            n_samples,n_protos = cost.size()
            if n_samples == 0:
                continue
          
            ratios = [1.0 / self.num_proto_per_type for _ in range(self.num_proto_per_type)]
            proto_constraint = torch.as_tensor(ratios, dtype=torch.float32, device=cost.device) * n_samples
            sample_constraint = torch.ones(n_samples, dtype=torch.float32, device=cost.device)
         
            
            _, assignment = SinkhornDistance(max_iter=self.max_iter).forward(
                sample_constraint, proto_constraint, cost)
           
            indexes = torch.argmax(assignment, dim=-1)
            onehot_indexes = F.one_hot(indexes, assignment.size(-1)).float()
    
                # 依据匹配选定target
            well_pred_mask_k = well_pred_mask[mask_k]        # gt为lable的span，是否预测正确

            well_pred_proto_mask_k = onehot_indexes * einops.repeat(
                well_pred_mask_k, "n -> n p", p=self.num_proto_per_type
            )       # 预测正确且为lable的span的原型匹配矩阵
            hiddens_k = hiddens[mask_k]                     # gt为k的span的表征
            well_pred_hiddens_k = hiddens_k * well_pred_mask_k.unsqueeze(-1)
            # 每个原型匹配到的表征的sum
            matched_hidden_sum = torch.einsum("np,nd->pd", well_pred_proto_mask_k, well_pred_hiddens_k)
            match_cnt = torch.sum(well_pred_proto_mask_k, dim=0)        # 每个原型匹配到多少个

            # 将target设置为匹配的结果
            target_indexes[mask_k] = indexes + self.num_proto_per_type * lable

            # 更新原型
            if torch.sum(match_cnt) > 0:
                update_mask = match_cnt != 0
                # normalize可以理解为一种mean
                matched_hidden_mean = F.normalize(matched_hidden_sum, p=2, dim=-1)
                newv = ema_update(protos[lable, update_mask], matched_hidden_mean[update_mask])
                protos[lable, update_mask] = newv
                
        # NOTE: 注意这里不能ddp，因为proto的更新没有走accelerate的流程，ddp需要特殊处理
        protos = einops.rearrange(protos, "k p d -> (k p) d")
        self.protos = torch.nn.Parameter(F.normalize(protos, dim=-1), requires_grad=False)

        return target_indexes

    # def compact_loss(self, logits, match_index):
    #         matched_logits = []
    #         matched_logits = logits[range(len(match_index)), match_index]
    #         loss = torch.pow(1 - matched_logits, 2)
    #         loss = torch.mean(loss)
    #         return loss, self.compact_weight
    
    def compact_loss(self, logits, match_index):
        matched_logits = []
        for logits_, index in zip(logits, match_index):
            matched_logits.append(logits_[index])
        if len(matched_logits) == 0:
            loss = 0
        else:
            matched_logits = torch.stack(matched_logits)
            # 希望相似度增加
            loss = torch.pow(1 - matched_logits, 2)
            loss = torch.mean(loss)
        return loss, self.compact_weight
    
    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        dec_table_ids=None,
        dec_table_attention_mask=None,
        dec_prompt_lens=None,
        trigger_enc_token_index=None,
        list_arg_slots=None,
        list_target_info=None,
        old_tok_to_new_tok_indexs=None,
        list_roles=None,
        list_arg_2_prompt_slots=None,
        cum_event_nums_per_type=None,
        list_dec_prompt_ids=None,
        list_len_prompt_ids=None
    ):
        """
        Args:
            multi args post calculation
        """
        enc_outputs = self.roberta(
            input_ids=enc_input_ids,
            attention_mask=enc_mask_ids,
            output_hidden_states=True,
            fully_encode=True
        ).hidden_states

        decoder_context = enc_outputs[self.decode_layer_start]
        if self.config.context_representation == 'decoder':
            context_outputs = enc_outputs[-1]
        else:
            context_outputs = decoder_context


        """ Transfer dec_table_ids into dec_table_embeds """
        input_shape = dec_table_ids.size()
        batch_size, table_seq_len = input_shape

        dec_table_embeds = torch.zeros((batch_size, table_seq_len, self.config.hidden_size),
                                        dtype=torch.float32, device=self.config.device)

        prompt_attention_mask = torch.zeros_like(list_dec_prompt_ids)
        for i, len_prompt_ids in enumerate(list_len_prompt_ids):
            prompt_attention_mask[i, :len_prompt_ids] = 1

        dec_prompt_embeds = self.roberta(
            input_ids=list_dec_prompt_ids,
            attention_mask=prompt_attention_mask,
            cross_attention=False
        ).last_hidden_state

        cusor = 0
        list_num_event_types = [len(x) for x in cum_event_nums_per_type]
        assert sum(list_num_event_types) == len(dec_prompt_embeds)
        for i, num_event_types in enumerate(list_num_event_types):
            assert sum(list_len_prompt_ids[cusor: cusor + num_event_types]) == dec_prompt_lens[i]
            cum_len = 0
            list_len_prompt_ids_ = list_len_prompt_ids[cusor: cusor + num_event_types]

            if self.num_prompt_pos > 0:
                pos = torch.arange(num_event_types, device=self.config.device)
                if self.training:
                    pos = torch.randperm(num_event_types, device=self.config.device)

            for j, len_prompt_ids in enumerate(list_len_prompt_ids_):
                dec_table_embeds[i, cum_len: cum_len + len_prompt_ids] = dec_prompt_embeds[cusor, :len_prompt_ids]
                if self.num_prompt_pos > 0:
                    dec_table_embeds[i, cum_len: cum_len + len_prompt_ids] += self.event_type_embed(pos[j])
                cum_len += len_prompt_ids
                cusor += 1

        # init arg slots' embeds with prompt slots' embeds
        for i, (list_arg_2_prompt_slots_, list_arg_slots_, cum_event_nums_per_type_) in \
            enumerate(zip(list_arg_2_prompt_slots, list_arg_slots, cum_event_nums_per_type)):
            dec_table_embeds_ = dec_table_embeds[i].detach()
            for j, arg_2_prompt_slots in enumerate(list_arg_2_prompt_slots_):
                event_index_start = cum_event_nums_per_type_[j-1] if j > 0 else 0
                event_index_end = cum_event_nums_per_type_[j]
                arg_slots = list_arg_slots_[event_index_start: event_index_end]
                for k, prompt_slots in enumerate(arg_2_prompt_slots.values()):
                    arg_slots_same_role = [arg_slot[k] for arg_slot in arg_slots]
                    for s, (start, end) in enumerate(zip(prompt_slots['tok_s'], prompt_slots['tok_e'])):
                        prompt_slot_embed = dec_table_embeds_[start: end]
                        prompt_slot_embed = torch.mean(prompt_slot_embed, dim=0)
                        arg_slots_same_cloumn = [arg_slot[s] for arg_slot in arg_slots_same_role]
                        dec_table_embeds[i, arg_slots_same_cloumn] = prompt_slot_embed

        if self.num_event_embed > 0:
            pos = torch.arange(self.num_event_embed, device=self.config.device)
            if self.training:
                pos = torch.randperm(self.num_event_embed, device=self.config.device)

        for i, (encoder_output, trigger_index, list_arg_slots_) in \
            enumerate(zip(decoder_context, trigger_enc_token_index, list_arg_slots)):
            
            dec_trigger_index = [arg_slots[0][0] - 1 for arg_slots in list_arg_slots_]
            assert len(trigger_index) == len(dec_trigger_index)
            for j, (trigger_start, trigger_end) in enumerate(trigger_index):
                # copy triggers' representation
                dec_trigger_index_ = dec_trigger_index[j]
                trigger_embed = encoder_output[trigger_start: trigger_end]
                trigger_embed = torch.mean(trigger_embed, dim=0)
                dec_table_embeds[i, dec_trigger_index_] = trigger_embed
                if self.num_event_embed > 0:
                    dec_table_embeds[i, dec_trigger_index_] += self.event_embed(pos[j])

                # add markers' representation to arg_slots
                arg_slots = list_arg_slots_[j]
                arg_slots = list(chain(*arg_slots))
                dec_table_embeds[i, arg_slots] += (encoder_output[trigger_start-1] + encoder_output[trigger_end]) / 2
                dec_table_embeds[i, arg_slots] /= 2

        # dec_table_embeds = self.dec_input_drop(dec_table_embeds)

        decoder_table_outputs = self.roberta(
                inputs_embeds=dec_table_embeds,
                attention_mask=dec_table_attention_mask,
                encoder_hidden_states=decoder_context,
                encoder_attention_mask=enc_mask_ids,
                cross_attention=True,
        )
        decoder_table_outputs = decoder_table_outputs.last_hidden_state   #[bs, table_seq_len, H]

        logit_lists = list()
        total_loss = 0.
        for i, (context_output, decoder_table_output, list_arg_slots_, list_roles_, old_tok_to_new_tok_index) in \
            enumerate(zip(context_outputs, decoder_table_outputs, list_arg_slots, list_roles, old_tok_to_new_tok_indexs)):
            
            batch_loss = list()
            cnt = 0
            target_roleids = []
            pred_roleids= []
            role_ids =[0]
            all_span_features = []
            list_output = list()
            # iterate event by event
            for j, (arg_slots, roles) in enumerate(zip(list_arg_slots_, list_roles_)):
                if self.training:
                    target_info = list_target_info[i][j]

                output = dict()
                for (slots, arg_role) in zip(arg_slots, roles):
                    role_ids.append(self.role2id[arg_role])
                    start_logits_list = list()
                    end_logits_list = list()
                    for slot in slots:
                        query_sub = decoder_table_output[slot].unsqueeze(0)
                        
                        start_query = (query_sub*self.w_prompt_start).unsqueeze(-1) # [1, H, 1]
                        end_query = (query_sub*self.w_prompt_end).unsqueeze(-1)     # [1, H, 1]

                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()  
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)
                    
                    output[arg_role] = [start_logits_list, end_logits_list]

            
                    if self.training:
                        # calculate loss
                        target = target_info[arg_role] # "arg_role": {"text": ,"span_s": ,"span_e": }
                        predicted_spans = list()
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index, self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()

                        span_features = []

                        for predicted_span in predicted_spans:
                            span_features.append(context_output[predicted_span[0]:predicted_span[1]+1].mean(0))


                        target_spans = [[s,e] for (s,e) in zip(target["span_s"], target["span_e"])]
                        if len(target_spans)<len(predicted_spans):
                            # need to consider whether to make more 
                            pad_len = len(predicted_spans) - len(target_spans)
                            target_spans = target_spans + [[0,0]] * pad_len
                            target["span_s"] = target["span_s"] + [0] * pad_len
                            target["span_e"] = target["span_e"] + [0] * pad_len
                            
                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                        else:
                            idx_preds = list(range(len(predicted_spans)))
                            idx_targets = list(range(len(target_spans)))
                            if len(idx_targets) > len(idx_preds):
                                idx_targets = idx_targets[0:len(idx_preds)]
                            idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                            idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                        for idx_pred,idx_target in zip(idx_preds,idx_targets):
                            all_span_features.append(span_features[idx_pred])
                            if target["span_s"][idx_target] ==0:
                                target_roleids.append(self.role2id['None'])
                            else:
                                target_roleids.append(self.role2id[arg_role])
                            pred_roleids.append(self.role2id[arg_role])

                        cnt += len(idx_preds)
                        start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds], torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                        end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds], torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                        batch_loss.append((start_loss + end_loss)/2)

                list_output.append(output)

            logit_lists.append(list_output)   
           
            if self.training:
                protos = self.protos.clone().cuda() 
                role_in_proto_ids  = [i for r in role_ids for i in range(r*self.num_proto_per_type,r*self.num_proto_per_type+self.num_proto_per_type)]
                label_masks = torch.zeros((protos.size(0)))
                label_masks[role_in_proto_ids] = 1
                all_span_features = torch.stack(all_span_features)

                pro_sem = torch.matmul(F.normalize(all_span_features, dim=-1), F.normalize(protos, dim=-1).transpose(-1, -2))       
                label_masks_expand = label_masks.unsqueeze(0).expand(all_span_features.size(0), -1).cuda()
                pro_sem = pro_sem.masked_fill(label_masks_expand == 0, -1e4)
                pred_tags = torch.div(torch.argmax(pro_sem, dim=-1), self.num_proto_per_type, rounding_mode="floor")
                pred_tags = torch.tensor(pred_tags).cuda()
                
                if self.training:
                    total_loss = total_loss + torch.sum(torch.stack(batch_loss))/cnt 
                    match_index = self.sinkhorn_matching_and_ema(
                            hiddens = all_span_features,
                            logits =  pro_sem,
                            pred_tags = torch.tensor(pred_tags).cuda(),
                            gts = torch.tensor(target_roleids).cuda(), # target_roleids  pred_roleids
                            role_ids = role_ids,
                    )       
                    ce_loss = CrossEntropyLoss()
                    label_loss = ce_loss(pro_sem, match_index)   
                    compact_loss, compact_weight = self.compact_loss(pro_sem, match_index)
        
                    proto_loss = label_loss+compact_weight*compact_loss
                    total_loss += proto_loss

        if self.training:
            return total_loss/len(context_outputs), logit_lists
        else:
            return [], logit_lists
