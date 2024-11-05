# SynDepNerAT
Transfer adversarial model for dependency parsing and NER tasks

The shared encoding layer reads the dependency parsing dataset and the NER dataset to complete word embedding, and after encoding, it is superimposed with the word vectors output by the private encoding layer of each task, thereby achieving information sharing between tasks.
