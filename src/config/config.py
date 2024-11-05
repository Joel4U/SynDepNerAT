# 
# @author: Allan
#

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Any
# from src.common import Instance
import torch
from enum import Enum
import os

class PaserModeType(Enum):
    biaf = 0
    span = 1

_dim_map = {
    "concat": 1,
    "gate-concat": 2,
}

class TaskType(Enum):
    ner = 0
    dep = 1

class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """

        # Model hyper parameters
        self.embedding_file = args.embedding_file if "embedding_file" in args.__dict__ else None
        self.embedding_dim = args.embedding_dim if "embedding_dim" in args.__dict__ else None
        self.context_emb_size = 0
        self.embedding, self.embedding_dim = self.read_pretrain_embedding() if "embedding_file" in args.__dict__ else (None, None)
        self.word_embedding = None
        self.seed = args.seed

        self.use_brnn = True
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn if "use_char_rnn" in args.__dict__ else None

        self.embedder_type = args.embedder_type if "embedder_type" in args.__dict__ else None
        self.embedder_freezing = args.embedder_freezing
        self.pos_embed_dim = args.pos_embed_dim
        self.emb_dropout = args.emb_dropout
        # ner Data specification
        self.train_nerfile = "data/" + args.ner_dataset + "/train.txt"
        self.dev_nerfile = "data/" + args.ner_dataset + "/dev.txt"
        self.test_nerfile = "data/" + args.ner_dataset + "/test.txt"
        # dep Data specification
        self.train_depfile = "data/" + args.dep_dataset + "/train.txt"
        self.dev_depfile = "data/" + args.dep_dataset + "/dev.txt"
        self.test_depfile = "data/" + args.dep_dataset + "/test.txt"

        # Training hyperparameter
        self.train_with_dev = args.train_with_dev
        self.other_lr_ner = args.other_lr_ner
        self.pretr_lr_ner = args.pretr_lr_ner
        self.other_lr_dep = args.other_lr_dep
        self.pretr_lr_dep = args.pretr_lr_dep
        self.weight_decay = args.weight_decay
        self.adv_weight = args.adv_weight

        self.enc_type = args.enc_type
        self.enc_nlayers = args.enc_nlayers
        self.enc_dropout = args.enc_dropout
        self.enc_dim = args.enc_dim

        self.num_epochs = args.num_epochs
        self.depdata_bz = args.depdata_bz
        self.nerdata_bz = args.nerdata_bz
        self.device = torch.device(args.device) if "device" in args.__dict__ else None
        self.max_no_incre = args.max_no_incre
        self.max_grad_norm = args.max_grad_norm if "max_grad_norm" in args.__dict__ else None
        # dep
        self.mlp_arc_dim = args.mlp_arc_dim
        self.mlp_rel_dim = args.mlp_rel_dim
        self.depbiaf_dropout = args.depbiaf_dropout
        # ner
        self.biaf_out_dim = args.biaf_out_dim
        self.sb_epsilon = args.sb_epsilon
        self.sb_size = 1            # Boundary smoothing window size
        self.sb_adj_factor = 1      # Boundary smoothing probability adjust factor
        self.sl_epsilon = 0         # Label smoothing loss epsilon
        self.ner_parser_mode = PaserModeType[args.ner_parser_mode]

        self.shared_input_dim = args.shared_input_dim
        self.shared_enc_type = args.shared_enc_type
        self.shared_enc_nlayers = args.shared_enc_nlayers
        self.shared_enc_dropout = args.shared_enc_dropout
        self.fusion_dropout = args.fusion_dropout
        self.fusion_type = args.fusion_type
        self.advloss_dropout = args.advloss_dropout
        self.concat_dropout = args.concat_dropout

        self.earlystop_atr = args.earlystop_atr if "earlystop_atr" in args.__dict__ else None


    def read_pretrain_embedding(self) -> Tuple[Union[Dict[str, np.array], None], int]:
        """
        Read the pretrained word embeddings, return the complete embeddings and the embedding dimension
        :return:
        """
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        else:
            exists = os.path.isfile(self.embedding_file)
            if not exists:
                print("[Warning] pretrain embedding file not exists, using random embedding",  'red')
                return None, self.embedding_dim
                # raise FileNotFoundError("The embedding file does not exists")
        embedding_dim = -1
        embedding = dict()
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    # print(tokens)
                    # print(embedding_dim)
                    assert (embedding_dim + 1 == len(tokens))
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                first_col = tokens[0]
                embedding[first_col] = embedd
        return embedding, embedding_dim



    def build_emb_table(self, word2idx: Dict[str, int]) -> None:
        """
        build the embedding table with pretrained word embeddings (if given otherwise, use random embeddings)
        :return:
        """
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(word2idx), self.embedding_dim])
            for word in word2idx:
                if word in self.embedding:
                    self.word_embedding[word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[word2idx[word], :] = self.embedding[word.lower()]
                else:
                    # self.word_embedding[self.word2idx[word], :] = self.embedding[self.UNK]
                    self.word_embedding[word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None  ## remove the pretrained embedding to save memory.
        else:
            self.word_embedding = np.empty([len(word2idx), self.embedding_dim])
            for word in word2idx:
                self.word_embedding[word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

