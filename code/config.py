import argparse
import json
import os

import numpy as np

data_dir = '../data/'
prepro_dir = os.path.join(data_dir, 'prepro_data/')
if not os.path.exists(prepro_dir):
    os.mkdir(prepro_dir)

rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), "r"))
id2rel = {v: k for k, v in rel2id.items()}
word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), "r"))
ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), "r"))
# word2vec = np.load(os.path.join(data_dir, 'vec.npy')) # 单词到向量的映射表


def get_opt():
    parser = argparse.ArgumentParser()

    # datasets path
    parser.add_argument('--train_set', type=str, default=os.path.join(data_dir, 'train.json'))
    parser.add_argument('--dev_set', type=str, default=os.path.join(data_dir, 'dev.json'))
    parser.add_argument('--test_set', type=str, default=os.path.join(data_dir, 'test.json'))

    # save path of preprocessed datasets
    parser.add_argument('--train_set_save', type=str, default=os.path.join(prepro_dir, 'train.pkl'))
    parser.add_argument('--dev_set_save', type=str, default=os.path.join(prepro_dir, 'dev.pkl'))
    parser.add_argument('--test_set_save', type=str, default=os.path.join(prepro_dir, 'test.pkl'))
    parser.add_argument('--dev_period', type=int, default=2)
    
    # checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--fig_result_dir', type=str, default='fig_result')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')

    # task/Dataset-related
    parser.add_argument('--vocabulary_size', type=int, default=200000)
    parser.add_argument('--relation_num', type=int, default=97)
    parser.add_argument('--entity_type_num', type=int, default=8)
    parser.add_argument('--max_entity_num', type=int, default=80)

    # padding
    parser.add_argument('--word_pad', type=int, default=0)
    parser.add_argument('--entity_type_pad', type=int, default=0)
    parser.add_argument('--entity_id_pad', type=int, default=0)

    # word embedding
    parser.add_argument('--word_emb_size', type=int, default=10)
    parser.add_argument('--pre_train_word', action='store_true')
    parser.add_argument('--data_word_vec',  type=str)  # 我尝试在这里放上default 值，但是因为要用json.dumps(...) 报错，所以这里就取消了
    parser.add_argument('--finetune_word', action='store_true')

    # entity type embedding
    parser.add_argument('--use_entity_type', action='store_true')
    parser.add_argument('--entity_type_size', type=int, default=36) # 是否使用entity_type 这个参数，如果使用的话，就将其设置成20的embedding

    # entity id embedding, i.e., coreference embedding in DocRED original paper
    parser.add_argument('--use_entity_id', action='store_true')
    parser.add_argument('--entity_id_size', type=int, default=20)

    # BiLSTM
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--lstm_hidden_size', type=int, default=100)
    parser.add_argument('--lstm_dropout', type=float, default=0.1)

    # training settings
    
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    

    # gcn
    parser.add_argument('--mention_drop', action='store_true')
    parser.add_argument('--gcn_layers', type=int, default=2)
        

    # BERT
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--bert_path', type=str, default="")
    parser.add_argument('--bert_fix', action='store_true')
    parser.add_argument('--coslr', action='store_true')
    parser.add_argument('--clip', type=float, default=-1)

    parser.add_argument('--k_fold', type=str, default="none")

    # use BiLSTM / BERT encoder, default: BiLSTM encoder
    parser.add_argument('--use_model', type=str, default="bert", choices=['bilstm', 'bert'],
                        help='you should choose between bert and bilstm')

    # binary classification threshold, automatically find optimal threshold when -1
    parser.add_argument('--input_theta', type=float, default=-1)
    # parser.add_argument('--use_global_graph', default=True)
    parser.add_argument('--alpha', type=float,default=0.00) # 生成prior attention 时的权重比例
    parser.add_argument('--beta', type=float,default=0.5) 
    parser.add_argument('--kb_path', type=str,default=None)  # 加载kb的路径
    parser.add_argument('--threshold',type=float ,default=0.1)  # 用于控制
    parser.add_argument('--fig_file_name',type=str)          
    parser.add_argument('--interaction_hidden_dropout_prob', type=float,default=0.1) # 后几层 Transformer 的交互中的dropout 比例
    parser.add_argument('--relation_dim', type=int,default=97) # relation embedding 的维度

    parser.add_argument('--interaction_layers', type=int,default=3) # 实体交互的transformer 层数
    parser.add_argument('--pair_interaction_layers', type=int,default=1) # 实体对交互的transformer 层数
    parser.add_argument('--t_head_num', type=int,default=8) # 交互层transformer 中的头数
    parser.add_argument('--graph_threshold',type=float,default=0.95)  # 利用预测得到的关系进行建图，这个关系的概率    
    parser.add_argument('--lr_rgcn', type=float,default=1e-3)  # rgcn 的学习率
    parser.add_argument('--lr_bert', type=float,default=2e-5)  # bert 的学习率    
    parser.add_argument('--lr_tran', type=float,default=1e-4)  # n层transformer 的学习率 
    parser.add_argument('--lr', type=float, default=1e-3) # 线性predict 的学习率
    parser.add_argument('--lr_bias', type=float, default=1e-1) # n*n*rel 的bias 的学习率
    parser.add_argument('--lr_word_embedding', type=float, default=1e-5) 
    parser.add_argument('--lr_weight', type=float, default=1e-2) 
    
    # 理解一下这个参数的作用
    # parser.add_argument('--gcn_dim', type=int, default=808)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--activation', type=str, default="relu")
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=30)
    
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--negativa_alpha', type=float, default=-1)  # negative example nums v.s positive example num
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--bert_dropout', type=float, default=0.1)    # dropout in bert
    parser.add_argument('--pos_lsr', type=float, default=1.0)  # label smoothing regularization for positive sample
    parser.add_argument('--neg_lsr', type=float, default=0.65)  # label smoothing regularization for negative sample
    parser.add_argument('--cs_weight', type=float, default=0.07)  # common sense 的权重系数
    
    parser.add_argument('--base_num', type=int, default=30)  # 基分解中base的个数
    parser.add_argument('--gcn_in_feat', type=int, default=808)  # 基分解中base的个数
    parser.add_argument('--gcn_out_feat', type=int, default=808)  # 基分解中base的个数

    parser.add_argument('--pos_weight', type=float,default=1.0) # 正样本的损失权重
    parser.add_argument('--neg_weight', type=float,default=1.0) # 负样本的损失权重
    parser.add_argument('--loss_type_weight', type=float,default=1e-6) # loss_type 的损失权重
    parser.add_argument('--notes', type=str) # 便于查看更改的备注
    parser.add_argument('--window', type=int, default=100)  # attention 中 window的大小
    parser.add_argument('--mask_prob', type=float, default=0.035)  # dataloader中加载数据时，mask掉实体的概率
    parser.add_argument('--T_max', type=int, default=71)  # 使用cosAnnealingLR 的参数值

    return parser.parse_args()
