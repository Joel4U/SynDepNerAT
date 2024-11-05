from tqdm import tqdm
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch
import json
import numpy as np
from src.data import Instance
from src.data.data_utils import root_dep_label, build_pos_idx, build_spanlabel_idx

from logger import get_logger



logger = get_logger()

PAD_INDEX = 0
PUNCT_LABEL = 'punct'  # EN 依存关系标签中的标点符号标签
# PUNCT_LABEL = 'P'  # CN 依存关系标签中的标点符号标签

def _spans_from_surrounding(span: Tuple[int], distance: int, num_tokens: int):
    """Spans with given `distance` to the given `span`.
    """
    for k in range(distance):
        for start_offset, end_offset in [(-k, -distance+k),
                                         (-distance+k, k),
                                         (k, distance-k),
                                         (distance-k, -k)]:
            start, end = span[0]+start_offset, span[1]+end_offset
            if 0 <= start < end <= num_tokens:
                yield (start, end)

def convert_instances_to_feature_tensors(instances: List[Instance], tokenizer: PreTrainedTokenizerFast, label2idx: Dict[str, int]) -> List[Dict]:
    features = []
    # print("[Data Info] We are not limiting the max length in tokenizer. You should be aware of that")
    for idx, inst in enumerate(instances):
        words = inst.ori_words
        orig_to_tok_index = []
        res = tokenizer.encode_plus(words, is_split_into_words=True)
        subword_idx2word_idx = res.word_ids(batch_index=0)
        prev_word_idx = -1
        for i, mapped_word_idx in enumerate(subword_idx2word_idx):
            if mapped_word_idx is None: ## cls and sep token
                continue
            if mapped_word_idx != prev_word_idx:
                orig_to_tok_index.append(i)
                prev_word_idx = mapped_word_idx
        assert len(orig_to_tok_index) == len(words)

        # tags = inst.pos
        # tag_ids = [pos2idx[tag] for tag in tags] if tags else [-100] * len(words)

        spanlabel_ids = {(spanlabel[1], spanlabel[2]): label2idx[spanlabel[0]] for spanlabel in inst.span_labels}

        features.append({"input_ids": res["input_ids"], "attention_mask": res["attention_mask"],
                         "orig_to_tok_index": orig_to_tok_index,# "tag_ids": tag_ids,
                         "spanlabel_ids": spanlabel_ids})
    return features


def ner_batch_variable(batch:List[Dict], config):
    device = config.device
    batch_size = len(batch)
    word_seq_lens = [len(feature["orig_to_tok_index"]) for feature in batch]
    max_seq_len = max(word_seq_lens)
    max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])

    input_ids = torch.full((batch_size, max_wordpiece_length), config.tokenizer.pad_token_id, dtype=torch.long, device=device)

    input_mask = torch.zeros((batch_size, max_wordpiece_length), dtype=torch.long, device=device)
    orig_to_tok_index = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    # pos_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    if config.sb_epsilon <= 0:
        spanlabel_ids_tensor = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.long, device=device)
    else:
        spanlabel_ids_tensor = torch.zeros((batch_size, max_seq_len, max_seq_len, len(config.nerlabel2idx)),
                                           dtype=torch.float32, device=device)
    for i, feature in enumerate(batch):
        input_ids[i, :len(feature["input_ids"])] = torch.tensor(feature["input_ids"], dtype=torch.long, device=device)
        input_mask[i, :len(feature["attention_mask"])] = torch.tensor(feature["attention_mask"], dtype=torch.long, device=device)
        orig_to_tok_index[i, :len(feature["orig_to_tok_index"])] = torch.tensor(feature["orig_to_tok_index"], dtype=torch.long, device=device)
        # pos_ids[i, :len(feature["tag_ids"])] = torch.tensor(feature["tag_ids"], dtype=torch.long, device=device)

        if config.sb_epsilon <= 0:
            spanlabel_ids = np.zeros((max_seq_len, max_seq_len), dtype=np.int64)
            for span, label in feature["spanlabel_ids"].items():
                start, end = span
                spanlabel_ids[start, end - 1] = label
        else:
            spanlabel_ids = np.zeros((max_seq_len, max_seq_len, len(config.nerlabel2idx)), dtype=np.float32)
            for span, label in feature["spanlabel_ids"].items():
                start, end = span
                spanlabel_ids[start, end - 1, label] += (1 - config.sb_epsilon)
                for dist in range(1, config.sb_size + 1):
                    eps_per_span = config.sb_epsilon / (config.sb_size * dist * 4)
                    sur_spans = list(_spans_from_surrounding((start, end), dist, max_seq_len))
                    for sur_start, sur_end in sur_spans:
                        spanlabel_ids[sur_start, sur_end - 1, label] += (eps_per_span * config.sb_adj_factor)
                    spanlabel_ids[start, end - 1, label] += eps_per_span * (dist * 4 - len(sur_spans))

            # 处理溢出的标签
            overflow_indic = np.sum(spanlabel_ids, axis=-1) > 1
            if np.any(overflow_indic):
                overflow_indices = np.where(overflow_indic)
                for j, k in zip(*overflow_indices):
                    spanlabel_ids[j, k] /= np.sum(spanlabel_ids[j, k])

            spanlabel_ids[:, :, config.nerlabel2idx['O']] = 1 - np.sum(spanlabel_ids, axis=-1)

        # 将 numpy 转换为 tensor
        if config.sb_epsilon <= 0:
            spanlabel_ids_tensor[i, :max_seq_len, :max_seq_len] = torch.tensor(spanlabel_ids, dtype=torch.long, device=device)
        else:
            spanlabel_ids_tensor[i, :max_seq_len, :max_seq_len, :] = torch.tensor(spanlabel_ids, dtype=torch.float32, device=device)

    return {
        "input_ids": input_ids,  "attention_mask": input_mask,
        "orig_to_tok_index": orig_to_tok_index, # "pos_ids":  pos_ids,
        "word_seq_lens": torch.tensor(word_seq_lens, dtype=torch.long, device=device),
        # "dephead_ids": head_ids, "deplabel_ids": rel_ids,
        # "punc_mask": punc_mask,
        "spanlabel_ids": spanlabel_ids_tensor,
    }


class NERDataset(Dataset):

    def __init__(self, file: str, tokenizer: PreTrainedTokenizerFast, sb_epsilon: float,  is_train: bool = True,
                 label2idx: Dict[str, int] = None, pos2idx: Dict[str, int] = None, is_json: bool = False):
        ## read all the instances. sentences and labels
        self.sb_epsilon = sb_epsilon
        self.max_entity_length = 0
        self.total_entities = 0
        self.entity_length_counts = {}
        insts = self.read_from_json_clue(file=file) if is_json else self.read_file(file=file)
        self.insts = insts
        if is_train:
            if label2idx is not None:
                print(f"YOU ARE USING EXTERNAL label2idx, WHICH IS NOT BUILT FROM TRAINING SET.")
                self.label2idx = label2idx
            else:
                print(f"[Data Info] Using the training set to build label index")
                idx2label, label2idx = build_spanlabel_idx(self.insts)
                self.idx2labels = idx2label
                self.label2idx = label2idx
                # _, self.pos2idx = build_pos_idx(self.insts)
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            # self.pos2idx = pos2idx      # pos tags not in the ace2004, ace2005, resume et,al.
        self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx)
        self.tokenizer = tokenizer

    def read_from_json_clue(self, file:str)-> List[Instance]:
        print(f"[Data Info] Reading file: {file}")
        insts = []
        with open(file, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                chunks = []
                words = data["text"]
                for entity_type, entity_data in data["label"].items():
                    for _, span in entity_data.items():
                        chunks.append(((span[0][0], span[0][1]), entity_type))
                        chunks_len = span[0][1] - span[0][0]
                        self.entity_length_counts[chunks_len] = self.entity_length_counts.get(chunks_len, 0) + 1
                        self.total_entities += 1
                insts.append(Instance(words=words, ori_words=words, dep_heads=None, dep_labels=None, span_labels=chunks, labels=None))
        return insts

    def get_chunk_type(self, tok):
        tag_class = tok.split('-')[0]
        tag_type = '-'.join(tok.split('-')[1:])
        return tag_class, tag_type

    def get_chunks(self, seq):
        default = 'O'
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                if (i - chunk_start) > self.max_entity_length:
                    self.max_entity_length = (i - chunk_start)
                chunk_type, chunk_start = None, None
            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    if (i - chunk_start) > self.max_entity_length:
                        self.max_entity_length = (i - chunk_start)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            if len(seq) - chunk_start > self.max_entity_length:
                self.max_entity_length = len(seq) - chunk_start
            chunks.append(chunk)

        return chunks

    def read_file(self, file: str) -> List[Instance]:
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = ['<'+root_dep_label.lower()+'>']
            ori_words = ['<'+root_dep_label.lower()+'>']
            tags = [root_dep_label]
            labels = ['O']
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    continue
                if line == "" and len(words) > 1:
                    chunks = self.get_chunks(labels)
                    if 'msra' in file or 'Weibo' in file or 'resume' in file:
                        insts.append(Instance(words=words, ori_words=ori_words, pos=None, depheads=None, deplabels=None, span_labels=chunks))
                    else:
                        insts.append(Instance(words=words, ori_words=ori_words, pos=tags, depheads=None, deplabels=None, span_labels=chunks))

                    words = ['<'+root_dep_label.lower()+'>']
                    ori_words = ['<'+root_dep_label.lower()+'>']
                    tags = [root_dep_label]
                    # depheads = [0]
                    # deplabels = [root_dep_label]
                    labels = ['O']
                    continue
                elif line == "" and len(words) <= 1:
                    continue
                ls = line.split()
                if 'msra' in file or 'Weibo' in file or 'resume' in file:
                    word, label = ls[0], ls[-1]
                elif 'conll' in file:
                    word, tag, label = ls[0], ls[1], ls[-1]
                    tags.append(tag)
                else:
                    word, tag, label = ls[1], ls[3], ls[-1]
                    tags.append(tag)
                ori_words.append(word)
                words.append(word)
                labels.append(label)

        print(f"{len(insts)}, number of sentences in file: {file}, we had added 'root' word at the beginning of each sentence")
        return insts

    def get_max_token_len(self):
        return max(len(inst['orig_to_tok_index']) for inst in self.insts_ids)

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]
