import argparse
from src.config import Config
import time
from src.model import PreBiafAT
import torch
from src.config.utils import get_huggingface_optimizer_and_scheduler, detect_overlapping_level
from tqdm import tqdm
from src.data import DEPDataset, NERDataset, ner_batch_variable, dep_batch_variable
from src.data.data_utils import batch_iter
from src.config.metrics import precision_recall_f1_report
from src.config.eval import calc_evalu_acc
from transformers import set_seed, AutoTokenizer
from logger import get_logger
from termcolor import colored

logger = get_logger()

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:3", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=44, help="random seed")
    parser.add_argument('--ner_dataset', type=str, default="conll03")
    parser.add_argument('--dep_dataset', type=str, default="ptb")
    parser.add_argument('--nerdata_bz', type=int, default=48, help="default NER batch size is 10")
    parser.add_argument('--depdata_bz', type=int, default=48, help="default DEP batch size is 10")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")
    parser.add_argument('--train_with_dev', type=bool, default=False, help="whether to train with development set")
    parser.add_argument('--max_no_incre', type=int, default=40, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=5, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")
    ##model hyperparameter
    parser.add_argument('--other_lr_ner', type=str, default=1e-3, help="between 1e-3 and 3e-3")
    parser.add_argument('--pretr_lr_ner', type=float, default=2e-5, help="for ner")
    parser.add_argument('--other_lr_dep', type=str, default=2e-4, help="between 1e-4 and 3e-4")
    parser.add_argument('--pretr_lr_dep', type=float, default=2e-5, help="for ner")
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--adv_weight', type=float, default=0.07)

    parser.add_argument('--embedder_type', type=str, default="roberta-base", help="you can use 'bert-large-cased，bert-base-multilan-cased' and so on")
    parser.add_argument('--emb_dropout', type=float, default=0.5, help="dropout for SynLSTM or LSTM")
    parser.add_argument('--embedder_freezing', type=bool, default=True, help="you can freeze the word embedder")
    
    parser.add_argument('--pos_embed_dim', type=int, default=48, help='pos_tag embedding size, heads | pos_embed_dim')
    parser.add_argument('--enc_type', type=str, default='lstm', choices=['lstm', 'naivetrans', 'adatrans'], help='type of encode for dep and ner')
    parser.add_argument('--enc_nlayers', type=int, default=1, help='number of encoder layers, 3 for LSTM or 6 for Transformer')
    parser.add_argument('--enc_dropout', type=float, default=0.5, help='dropout used in transformer or lstm')
    parser.add_argument('--enc_dim', type=int, default=400, help="hidden size of the encoder, usually we set to 200 for LSTM, 512 for transformer (d_model)")

    parser.add_argument('--mlp_arc_dim', default=500, type=int, help='size of pos mlp hidden layer, dep')
    parser.add_argument('--mlp_rel_dim', default=100, type=int, help='size of xpos mlp hidden layer, dep')
    parser.add_argument('--depbiaf_dropout', default=0.33, type=float, help='dropout probability of biaffine, dep')

    parser.add_argument('--sb_epsilon', type=float, default=0.1, help="Boundary smoothing loss epsilon")
    parser.add_argument('--biaf_out_dim', type=int, default=150, help="hidden size of biaf ner")
    parser.add_argument('--ner_parser_mode', type=str, default="biaf", choices=["biaf", "span"], help="parser model consists of biaf or span for ner")

    parser.add_argument('--shared_input_dim', type=int, default=1024)
    parser.add_argument("--shared_enc_type", default='naivetrans', choices=['naivetrans', 'adatrans'])
    parser.add_argument("--shared_enc_nlayers", type=int, default=3, help='the number of shared encoder layer')
    parser.add_argument('--shared_enc_dropout', type=float, default=0.5)
    parser.add_argument('--fusion_type', type=str, default='gate-concat', choices=['concat', 'add', 'concat-add', 'gate-add', 'gate-concat'])
    parser.add_argument('--fusion_dropout', type=float, default=0.4)
    parser.add_argument('--advloss_dropout', type=float, default=0.8)
    parser.add_argument('--concat_dropout', type=float, default=0.5)


    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args

def train_model(config, epoch, train_nerdata, dev_nerdata, test_nerdata, train_depdata, dev_depdata=None, test_depdata=None):

    print(colored(f"[Info] number of training ner instances: {len(train_nerdata)}, dep instances: {len(train_depdata)}", "yellow"))
    print(colored(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}", 'red'))
    model = PreBiafAT(config)
    optimizer_ner, scheduler_ner = get_huggingface_optimizer_and_scheduler(model=model, pretr_lr=config.pretr_lr_ner, other_lr=config.other_lr_ner,
                                                                   num_training_steps=len(train_nerdata) * epoch,
                                                                   weight_decay=config.weight_decay, eps=1e-8,
                                                                   warmup_step=int(0.2 * len(train_nerdata) // config.nerdata_bz * epoch))
    # optimizer_dep, scheduler_dep = get_huggingface_optimizer_and_scheduler(model=model, pretr_lr=config.pretr_lr_dep, other_lr=config.other_lr_dep,
    #                                                             num_training_steps=len(train_depdata) * epoch,
    #                                                             weight_decay=config.weight_decay, eps=1e-8,
    #                                                             warmup_step=int(0.2 * len(train_depdata) // config.depdata_bz * epoch))
    # print(optimizer)
    model.to(config.device)
    best_dev = [-1, 0]
    best_test = [-1, 0]
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))
    for i in range(1, epoch + 1):
        epoch_nerloss, epoch_deploss, epoch_advloss = 0, 0, 0
        start_time = time.time()
        model.train()
        initial_adv_weight = 0.005  # 初始较小的权重
        if i <= 5:
            adv_weight = 0.0
        else:
            adv_weight = min(initial_adv_weight * (i / 5), config.adv_weight)
        # dep parsing
        for iter, batch_data in enumerate(batch_iter(train_depdata, config.depdata_bz, True)):
            batcher = dep_batch_variable(batch_data, config)
            deploss, advloss = model('dep', batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"],
                         batcher["attention_mask"], batcher["pos_ids"], batcher["dephead_ids"], batcher["deplabel_ids"], None)
            epoch_deploss += deploss.item()
            epoch_advloss += advloss.item()
            loss = deploss + adv_weight * advloss
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer_ner.step()
            optimizer_ner.zero_grad()
            scheduler_ner.step()
        # ner
        for iter, batch_data in enumerate(batch_iter(train_nerdata, config.nerdata_bz, True)):
            batcher = ner_batch_variable(batch_data, config)
            nerloss, advloss = model('ner', batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"], batcher["attention_mask"],
                         None, None, None, batcher["spanlabel_ids"])
            epoch_nerloss += nerloss.item()
            epoch_advloss += advloss.item()
            loss = nerloss + adv_weight * advloss
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer_ner.step()
            optimizer_ner.zero_grad()
            scheduler_ner.step()
        
        end_time = time.time()
        print("\n")
        logger.info(f"Epoch {i}, NER Bert_lr: {scheduler_ner.get_last_lr()[0]:.4e}，NER Other_lr: {scheduler_ner.get_last_lr()[2]:.4e},\n "
                    # f"Epoch {i}, DEP Bert_lr: {scheduler_dep.get_last_lr()[0]:.4e}, DEP Other_lr: {scheduler_dep.get_last_lr()[2]:.4e},\n "
                    f"epoch_deploss: {epoch_deploss:.5f}, epoch_nerloss: {epoch_nerloss:.5f},  epoch_advloss: {epoch_advloss:.5f}, Time is {(end_time - start_time):.2f}s")

        model.eval()
        if dev_nerdata is not None:
            dev_metrics = evaluate_nermodel(config, model, dev_nerdata, "dev")
        test_metrics = evaluate_nermodel(config, model, test_nerdata, "test")
        if dev_depdata is not None:
            dev_uas, dev_las = evaluate_depmodel(config, model, dev_depdata, "dev")
            test_uas, test_las = evaluate_depmodel(config, model, test_depdata, "test")

        if test_metrics[2] > best_test[0]:
            no_incre_dev = 0
            # best_dev[0] = dev_metrics[2]
            # best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

def evaluate_nermodel(config, model, dataset, name):
    set_y_pred = []
    with torch.no_grad():
        for iter, batch_data in enumerate(batch_iter(dataset, config.nerdata_bz, False)):
            batcher = ner_batch_variable(batch_data, config)
            batch_y_pred = model('ner', batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"], batcher["attention_mask"], 
                                 None, None, None, batcher["spanlabel_ids"], is_train=False)
            set_y_pred.extend(batch_y_pred)
        set_y_gold = [inst.span_labels for inst in dataset.insts]
        _, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
        precise = ave_scores['micro']['precision']
        recall = ave_scores['micro']['recall']
        fscore = ave_scores['micro']['f1']
        logger.info(f"[NER {name}set Total] Prec.: { precise * 100:.2f}, Rec.: {recall * 100:.2f}, Micro F1: {fscore * 100:.2f}")
    return [precise, recall, fscore]

def evaluate_depmodel(config, model, dataset, name):
    total_arc_acc, total_rel_acc, total_valid_arcs = 0, 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for i, batch_data in enumerate(batch_iter(dataset, config.depdata_bz, True)):
            batcher = dep_batch_variable(batch_data, config)
            S_arc, S_rel = model('dep', batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"],
                                 batcher["attention_mask"], batcher["pos_ids"], None, None, None, is_train=False)
            batch_size, sent_len = batcher["orig_to_tok_index"].size()
            maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=batcher["orig_to_tok_index"].device).view(1, sent_len).expand(batch_size, sent_len)
            non_pad_mask = torch.le(maskTemp, batcher["word_seq_lens"].view(batch_size, 1).expand(batch_size, sent_len))
            arc_acc, rel_acc, valid_arc = calc_evalu_acc(S_arc, S_rel, batcher["dephead_ids"], batcher["deplabel_ids"], non_pad_mask, batcher["punc_mask"])
            total_arc_acc += arc_acc
            total_rel_acc += rel_acc
            total_valid_arcs += valid_arc

    UAS = total_arc_acc * 100. / total_valid_arcs
    LAS = total_rel_acc * 100. / total_valid_arcs
    logger.info(f"[DEP {name}set Total] UAS.: {UAS:4.2f}, LAS.: {LAS:4.2f}")
    return UAS, LAS

def main():
    parser = argparse.ArgumentParser(description="SynDepAT implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    logger.info(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer")
    conf.tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True, use_fast=True)
    print(colored(f"[Data Info] Reading dataset from: \t{conf.train_nerfile}\t{conf.dev_nerfile}\t{conf.test_nerfile}", "yellow"))
    train_nerdataset = NERDataset(conf.train_nerfile, conf.tokenizer, conf.sb_epsilon)
    # conf.pos_size = len(train_dataset.pos2idx)
    conf.nerlabel2idx = train_nerdataset.label2idx
    conf.idx2nerlabel = train_nerdataset.idx2labels
    conf.nerlabel_size = len(train_nerdataset.label2idx)

    dev_nerdataset = NERDataset(conf.dev_nerfile, conf.tokenizer, conf.sb_epsilon, label2idx=train_nerdataset.label2idx, is_train=False)
    test_nerdataset = NERDataset(conf.test_nerfile,  conf.tokenizer, conf.sb_epsilon, label2idx=train_nerdataset.label2idx, is_train=False)
    conf.max_entity_length = max(max(train_nerdataset.max_entity_length, dev_nerdataset.max_entity_length), test_nerdataset.max_entity_length)
    conf.max_seq_length = max(max(train_nerdataset.get_max_token_len(), dev_nerdataset.get_max_token_len()), test_nerdataset.get_max_token_len())
    all_insts = train_nerdataset.insts + dev_nerdataset.insts + test_nerdataset.insts
    conf.overlapping_level = max(detect_overlapping_level(inst.span_labels) for inst in all_insts)

    print(colored(f"[Data Info] Reading dataset from: \t{conf.train_depfile}\t{conf.dev_depfile}\t{conf.test_depfile}", "yellow"))
    train_depdataset = DEPDataset(conf.train_depfile, conf.tokenizer)
    conf.deplabel2idx = train_depdataset.deplabel2idx
    conf.deppos_size = len(train_depdataset.pos2idx)
    conf.rel_size = len(train_depdataset.deplabel2idx)
    conf.punctid = train_depdataset.punctid
    dev_depdataset = DEPDataset(conf.dev_depfile, conf.tokenizer, deplabel2idx=train_depdataset.deplabel2idx, pos2idx=train_depdataset.pos2idx, is_train=False)
    test_depdataset = DEPDataset(conf.test_depfile, conf.tokenizer, deplabel2idx=train_depdataset.deplabel2idx, pos2idx=train_depdataset.pos2idx, is_train=False)
    train_model(conf, conf.num_epochs, train_nerdataset, dev_nerdataset, test_nerdataset, train_depdataset, dev_depdataset, test_depdataset)

if __name__ == "__main__":
    main()
