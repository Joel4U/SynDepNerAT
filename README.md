# SynDepSrlAT
Transfer adversarial model for dependency parsing and NER tasks

The shared encoding layer reads the dependency parsing dataset and the NER dataset to complete word embedding, and after encoding, it is superimposed with the word vectors output by the private encoding layer of each task, thereby achieving information sharing between tasks.


# Parameters

embedder_type: roberta-base
optimizer: adamw
pretr_lr: 2e-05
other_lr: 0.001
max_grad_norm: 1.0
pred_embed_dim: 48
enc_dim: 400
mlp_dim: 300
LSTM(1)-Transformers(1)

# Performance
**CoNLL 2005 WSJ**

| Models  | P. | R. | F1 |
| ------------- | ------------- |------------- |------------- |
| BiLSTM-Span + ELMo + Ensemble | 89.2  |  87.9   |  88.5
| Pre-SRL + RoBERTa Base | 89.05  |  89.17  | 89.11
| SynDepSrlAT + RoBERTa Base | 88.59  | 89.26  | 88.92

**CoNLL 2005 Brown**

| Models  | P. | R. | F1 |
| ------------- | ------------- |------------- |------------- |
| BiLSTM-Span + ELMo + Ensemble| 81.0   |  78.4   | 79.6
| Pre-SRL + RoBERTa Base|   80.51   |  79.78 | 80.15
| SynDepSrlAT + RoBERTa Base | 80.36  |  79.74 | 80.05

**CoNLL 2012 EN**

| Models  | P. | R. | F1 |
| ------------- | ------------- |------------- |------------- |
| BiLSTM-Span + ELMo + Ensemble |  -    |  -    | 87.0
| Pre-SRL + RoBERTa Base |  87.27    | 88.12   | 87.69
| SynDepSrlAT + RoBERTa Base | 88.59  | 89.26  | 88.92

**PTB**

| Models  | UAS | LAS |
| ------------- | ------------- |------------- |
|  Deep Biaffine |  95.87 | 94.22	
| Pre-Biaf + RoBERTa Base|  96.60  | 95.21
| SynDepSrlAT + RoBERTa Base | 97.00   |  95.18

**CTB**

| Models  | UAS | LAS |
| ------------- | ------------- |------------- |
 | Deep Biaffine | 89.30 | 88.23  
|  Pre-Biaf + electra-base   |  90.66  | 87.07
| Pre-Biaf + bert-base-multi |  91.26  | 89.96
| SynDepSrlAT + bert-base-multi |   | 
