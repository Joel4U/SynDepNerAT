import torch
import torch.nn as nn

class FFNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, activation=nn.ReLU()):
        super(FFNEncoder, self).__init__()

        if activation is None:
            self.activation = lambda x: x
        else:
            assert callable(activation)
            self.activation = activation

        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        # self.reinit_layer_(self.linear, activation)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        linear_out = self.dropout(self.linear(inputs))
        return self.activation(linear_out)

    def reinit_layer_(self, layer: torch.nn.Module, nonlinearity='relu'):
        for name, param in layer.named_parameters():
            if name.startswith('bias'):
                torch.nn.init.zeros_(param.data)
            elif name.startswith('weight'):
                if nonlinearity.lower() in ('relu','leakyrelu' ):
                    torch.nn.init.kaiming_uniform_(param.data, nonlinearity=nonlinearity)
                elif nonlinearity.lower() in ('glu',):
                    torch.nn.init.xavier_uniform_(param.data, gain=torch.nn.init.calculate_gain('sigmoid'))
                else:
                    torch.nn.init.xavier_uniform_(param.data, gain=torch.nn.init.calculate_gain(nonlinearity))
            else:
                raise TypeError(f"Invalid Layer {layer}")

class BiaffineDecoder(nn.Module):
    def __init__(self, config, size_emb_dim: int = 25):
        super(BiaffineDecoder, self).__init__()
        self.max_span_width = config.max_entity_length
        self.out_dim = config.biaf_out_dim
        self.bias_x = True
        self.bias_y = True
        self.out_size = config.nerlabel_size
        self._act = nn.ReLU()
        # if config.activation == 'ReLU':
        #     self._act = nn.ReLU()
        # elif config.activation == 'LeakyReLU':
        #     self._act = nn.LeakyReLU(negative_slope=0.1)
        # elif config.activation == 'ELU':
        #     self._act = nn.ELU()
        # else:
        #     self._act = nn.Identity()
        self.affine_start = FFNEncoder(in_dim=config.enc_dim * 2, out_dim=self.out_dim, dropout=0, activation=self._act)
        self.affine_end = FFNEncoder(in_dim=config.enc_dim * 2, out_dim=self.out_dim, dropout=0, activation=self._act)
        self.size_embedding = nn.Embedding(self.max_span_width + 1, size_emb_dim)
        self.reinit_embedding_(self.size_embedding)
        # span_size_id
        self.register_buffer('_span_size_ids', torch.arange(config.max_seq_length) - torch.arange(config.max_seq_length).unsqueeze(-1))
        # Create `_span_non_mask` before changing values of `_span_size_ids`
        self.register_buffer('_span_non_mask', self._span_size_ids >= 0)
        self._span_size_ids.masked_fill_(self._span_size_ids < 0, 0)
        self._span_size_ids.masked_fill_(self._span_size_ids > config.max_entity_length, config.max_entity_length)
        # parameter：'U'，'W'， 'b'
        self.U = nn.Parameter(torch.empty(self.out_size, self.out_dim, self.out_dim))
        torch.nn.init.orthogonal_(self.U.data)
        # W and bias
        self.W = torch.nn.Parameter(torch.empty(self.out_size, self.out_dim*2 + size_emb_dim))
        self.b = torch.nn.Parameter(torch.empty(self.out_size))
        torch.nn.init.orthogonal_(self.W.data)
        torch.nn.init.zeros_(self.b.data)
        self.dropout = nn.Dropout(0.2)
        self.in_dropout = nn.Dropout(0.0)

    def reset_params(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def reinit_embedding_(self, embedding: nn.Embedding):
        uniform_range = (3 / embedding.weight.size(1)) ** 0.5

        torch.nn.init.uniform_(embedding.weight.data, -uniform_range, uniform_range)
        if embedding.padding_idx is not None:
            nn.init.zeros_(embedding.weight.data[embedding.padding_idx])

        print(
            "Embeddings initialized with randomized vectors \n"
            f"Vector average absolute value: {uniform_range / 2:.4f}"
        )

    def _get_span_size_ids(self, seq_len: int):
        return self._span_size_ids[:seq_len, :seq_len]

    def _get_span_non_mask(self, seq_len: int):
        return self._span_non_mask[:seq_len, :seq_len]

    def forward(self,  enc_feature):
        bz, seq_len, _ = enc_feature.size()
        affined_start = self.affine_start(self.in_dropout(enc_feature))
        affined_end = self.affine_end(self.in_dropout(enc_feature))

        scores1 = self.dropout(affined_start).unsqueeze(1).matmul(self.U).matmul(self.dropout(affined_end).permute(0, 2, 1).unsqueeze(1))
        # scores: (batch, start_seq_len, end_seq_len, label_size)
        scores = scores1.permute(0, 2, 3, 1)
        # affined_cat: (batch, start_seq_len, end_seq_len, affine_dim*2)
        affined_cat = torch.cat([self.dropout(affined_start).unsqueeze(2).expand(-1, -1, affined_end.size(1), -1),
                                 self.dropout(affined_end).unsqueeze(1).expand(-1, affined_start.size(1), -1, -1)], dim=-1)

        # size_embedded: (start_seq_len, end_seq_len, emb_dim)
        size_embedded = self.size_embedding(self._get_span_size_ids(seq_len))
        # affined_cat: (batch, start_seq_len, end_seq_len, affine_dim*2 + emb_dim)
        affined_cat = torch.cat([affined_cat, self.dropout(size_embedded).unsqueeze(0).expand(bz, -1, -1, -1)], dim=-1)

        # scores2: (label_size, affine_dim*2 + emb_dim) * (batch, start_seq_len, end_seq_len, affine_dim*2 + emb_dim, 1) -> (batch, start_seq_len, end_seq_len, label_size, 1)
        scores2 = self.W.matmul(affined_cat.unsqueeze(-1))
        # scores: (batch, start_seq_len, end_seq_len, label_size)
        scores = scores + scores2.squeeze(-1) + self.b
        return scores
        # return scores.contiguous()