import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.model.embedder import TransformersEmbedder
from src.model.module.transformer_enc import TransformerEncoder
from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.biaffine_decoder import BiaffineDecoder
from src.model.module.mlp_biaf import NonlinearMLP, Biaffine
from src.model.module.gate_fusion import FusionModule
from src.model.module.adv_loss import Adversarial_loss

from typing import Tuple, Union
from src.config.utils import _check_soft_target, _spans_from_upper_triangular, filter_clashed_by_priority, timestep_dropout
from src.config.config import _dim_map, TaskType


class PreBiafAT(nn.Module):
    def __init__(self, config):
        super(PreBiafAT, self).__init__()
        # common parameter
        self.device = config.device
        self.emb_dropout = config.emb_dropout
        self.idx2nerlabel = config.idx2nerlabel
        self.sb_epsilon = config.sb_epsilon
        self.overlapping_level = config.overlapping_level
        self.enc_type = config.enc_type
        self.concat_dropout = config.concat_dropout
        self.depbiaf_dropout = config.depbiaf_dropout
        # worde and pos embed
        self.word_embedder = TransformersEmbedder(transformer_model_name=config.embedder_type, is_freezing=config.embedder_freezing)
        self.tag_embedding = nn.Embedding(num_embeddings=config.deppos_size, embedding_dim=config.pos_embed_dim, padding_idx=0)
        self.transformer_drop = nn.Dropout(config.emb_dropout)
        # ner encode and decode
        input_dim = self.word_embedder.get_output_dim()
        if config.enc_type == 'adatrans' or config.enc_type == 'naivetrans':
            self.ner_context_encoder = TransformerEncoder(d_model=input_dim, num_layers=config.enc_nlayers, n_head=8,
                                              feedforward_dim=2 * input_dim, attn_type=self.enc_type, output_dim=config.enc_dim, dropout=config.enc_dropout)
        elif config.enc_type == 'lstm':
            self.ner_context_encoder = BiLSTMEncoder(input_dim=input_dim, hidden_dim=config.enc_dim, drop_lstm=config.enc_dropout, num_lstm_layers=config.enc_nlayers)
        
        self.biaffine_decoder = BiaffineDecoder(config)
        _span_size_ids = torch.arange(config.max_seq_length) - torch.arange(config.max_seq_length).unsqueeze(-1)
        self._span_non_mask = (_span_size_ids >= 0).to(torch.bool)

        self.ner_proj = nn.Linear(input_dim, config.shared_input_dim)
        # dep encode and decode
        input_dim = self.word_embedder.get_output_dim() + config.pos_embed_dim
        if config.enc_type == 'adatrans' or config.enc_type == 'naivetrans':
            self.dep_context_encoder = TransformerEncoder(d_model=input_dim, num_layers=config.enc_nlayers, n_head=8,
                                              feedforward_dim=2 * input_dim, attn_type=self.enc_type, output_dim=config.enc_dim, dropout=config.enc_dropout)
        elif config.enc_type == 'lstm':
            self.dep_context_encoder = BiLSTMEncoder(input_dim=input_dim, hidden_dim=config.enc_dim, drop_lstm=config.enc_dropout, num_lstm_layers=config.enc_nlayers)

        self._activation = nn.ReLU()
        self.mlp_arc = NonlinearMLP(in_feature=config.enc_dim * 2, out_feature=config.mlp_arc_dim * 2, activation=nn.ReLU())   # nn.LeakyReLU(0.1)
        self.mlp_rel = NonlinearMLP(in_feature=config.enc_dim * 2, out_feature=config.mlp_rel_dim * 2, activation=nn.ReLU())   # nn.LeakyReLU(0.1)
        self.arc_biaffine = Biaffine(config.mlp_arc_dim, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_dim, config.rel_size, bias=(True, True))

        self.dep_proj = nn.Linear(input_dim, config.shared_input_dim)

        # share encode and adverse
        # self.shared_encoder = TransformerEncoder(d_model=config.shared_input_dim, num_layers=config.shared_enc_nlayers, n_head=8,
                                                # feedforward_dim=2 * input_dim, attn_type=config.shared_enc_type, output_dim=config.enc_dim, dropout=config.shared_enc_dropout)
        self.shared_encoder = BiLSTMEncoder(input_dim=config.shared_input_dim, hidden_dim=config.enc_dim, drop_lstm=config.shared_enc_dropout, num_lstm_layers=config.shared_enc_nlayers)

        self.output_dim = config.enc_dim * _dim_map[config.fusion_type]
        self.fusion = FusionModule(fusion_type=config.fusion_type, layer=0, input_size=config.enc_dim,
                                   output_size=self.output_dim, dropout=config.fusion_dropout)
        # self.fusion = FusionModule(fusion_type=config.fusion_type, layer=0, input_size=config.enc_dim,
        #                            output_size=self.output_dim, dropout=config.fusion_dropout)
        self.adv_loss = Adversarial_loss(config.enc_dim, config.advloss_dropout)

    def forward(self, Task: str, subword_input_ids: torch.Tensor,  word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor, attention_mask: torch.Tensor, tag_ids: torch.Tensor,
                    trueheads: torch.Tensor,  truerels: torch.Tensor,
                    span_label_ids: torch.Tensor, is_train: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        bz, sent_len = orig_to_tok_index.size()
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=subword_input_ids.device).view(1, sent_len).expand(bz, sent_len)
        non_pad_mask = torch.le(maskTemp, word_seq_lens.view(bz, 1).expand(bz, sent_len))
        word_emb = self.word_embedder(subword_input_ids, orig_to_tok_index, attention_mask)
        if Task == 'dep':
            tags_emb = self.tag_embedding(tag_ids)
            word_rep = torch.cat((word_emb, tags_emb), dim=-1).contiguous()
            if self.training:
                word_rep = timestep_dropout(word_rep, self.emb_dropout)
            # private dep encoder
            if self.enc_type == 'lstm':
                private_enc_out = self.dep_context_encoder(word_rep, word_seq_lens)
            else:
                private_enc_out = self.dep_context_encoder(word_rep, non_pad_mask)
            # shared dep encoder
            shared_embed = F.dropout(self.dep_proj(word_rep))
            # shared_enc_out = self.shared_encoder(shared_embed, non_pad_mask)
            shared_enc_out = self.shared_encoder(shared_embed, word_seq_lens)
            concat_enc_out = self.fusion(private_enc_out, shared_enc_out)
            if is_train:
                concat_enc_out = timestep_dropout(concat_enc_out, self.concat_dropout)

            arc_feat = self.mlp_arc(concat_enc_out)
            rel_feat = self.mlp_rel(concat_enc_out)
            arc_head, arc_dep = arc_feat.chunk(2, dim=-1)
            rel_head, rel_dep = rel_feat.chunk(2, dim=-1)

            if is_train:
                arc_head = timestep_dropout(arc_head, self.depbiaf_dropout)
                arc_dep = timestep_dropout(arc_dep, self.depbiaf_dropout)
                rel_head = timestep_dropout(rel_head, self.depbiaf_dropout)
                rel_dep = timestep_dropout(rel_dep, self.depbiaf_dropout)

            S_arc = self.arc_biaffine(arc_dep, arc_head).squeeze(-1)
            S_rel = self.rel_biaffine(rel_dep, rel_head)
            if is_train:
                # task_dep = torch.ones((bz, 2), device=self.device, dtype=torch.int8)
                task_dep = torch.zeros((bz,), device=self.device, dtype=torch.long)
                adv_loss = self.adv_loss(shared_enc_out, task_dep)
                dep_loss = self.calc_deploss(S_arc, S_rel, trueheads, truerels, non_pad_mask)
                return dep_loss, adv_loss
            else:
                return S_arc, S_rel

        elif Task == 'ner':
            if self.training:
                word_rep = timestep_dropout(word_emb, self.emb_dropout)
            else:
                word_rep = word_emb
            # private ner encoder
            if self.enc_type == 'lstm':
                private_enc_out = self.ner_context_encoder(word_rep, word_seq_lens)
            else:
                private_enc_out = self.ner_context_encoder(word_rep, non_pad_mask)
            # shared ner encoder
            shared_embed = F.dropout(self.ner_proj(word_rep))
            # shared_enc_out = self.shared_encoder(shared_embed, non_pad_mask)
            shared_enc_out = self.shared_encoder(shared_embed, word_seq_lens)
            concat_enc_out = self.fusion(private_enc_out, shared_enc_out)
            if is_train:
                concat_enc_out = timestep_dropout(concat_enc_out, self.concat_dropout)

            biaffine_score = self.biaffine_decoder(concat_enc_out)
    
            if is_train:
                losses = []
                for curr_label_ids, curr_scores, curr_len in zip(span_label_ids, biaffine_score, word_seq_lens.cpu().tolist()):
                    curr_non_mask = self._get_span_non_mask(curr_len)
                    if self.sb_epsilon <=0:
                        loss = nn.functional.cross_entropy(curr_scores[:curr_len, :curr_len][curr_non_mask], curr_label_ids[:curr_len, :curr_len][curr_non_mask], reduction="sum")
                    else:
                        soft_target = curr_label_ids[:curr_len, :curr_len][curr_non_mask]
                        _check_soft_target(soft_target)
                        log_prob = curr_scores[:curr_len, :curr_len][curr_non_mask].log_softmax(dim=-1)
                        loss = -(log_prob * soft_target).sum(dim=-1)
                        loss = loss.sum()
                    losses.append(loss)
                # task_ner = torch.ones((bz, 2), device=self.device, dtype=torch.int8)
                task_ner = torch.ones((bz,), device=self.device, dtype=torch.long)
                adv_loss = self.adv_loss(shared_enc_out, task_ner)
                return torch.stack(losses).mean(), adv_loss
            else:
                batch_y_pred = []
                for curr_scores, curr_len in zip(biaffine_score, word_seq_lens.cpu().tolist()):
                    # curr_non_mask1 = curr_non_mask1[:curr_len, :curr_len]
                    curr_non_mask = self._get_span_non_mask(curr_len) 
                    confidences, label_ids = curr_scores[:curr_len, :curr_len][curr_non_mask].softmax(dim=-1).max(dim=-1)
                    labels = [self.idx2nerlabel[i] for i in label_ids.cpu().tolist()]
                    chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != 'O']  # '<none>'
                    confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != 'O']
                    assert len(confidences) == len(chunks)
                    # Sort chunks by confidences: high -> low
                    chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
                    chunks = filter_clashed_by_priority(chunks, allow_level=self.overlapping_level)  # self.overlapping_level: Flat 0, Nested 1, 这里就是对实体排序，选取策略的地方， 返回了个数是动态变化的
                    batch_y_pred.append(chunks)
                return batch_y_pred
        
    def _get_span_non_mask(self, seq_len: int):
        return self._span_non_mask[:seq_len, :seq_len]

    def calc_deploss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
        # non_pad_mask[:, 0] = 0  # mask out <root>
        # non_pad_mask = non_pad_mask.byte()
        pad_mask = (non_pad_mask == 0)

        bz, seq_len, _ = pred_arcs.size()
        masked_true_heads = true_heads.masked_fill(pad_mask, -1)
        arc_loss = nn.functional.cross_entropy(pred_arcs.reshape(bz*seq_len, -1), masked_true_heads.reshape(-1), ignore_index=-1)

        bz, seq_len, seq_len, rel_size = pred_rels.size()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_heads].contiguous()

        masked_true_rels = true_rels.masked_fill(pad_mask, -1)
        # (bz*seq_len, rel_size)  (bz*seq_len, )
        rel_loss = nn.functional.cross_entropy(out_rels.reshape(-1, rel_size), masked_true_rels.reshape(-1), ignore_index=-1)
        return arc_loss + rel_loss