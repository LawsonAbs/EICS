'''
添加 实体类型*实体类型*关系类型 标签
'''
import networkx as nx # 方便可视化图信息
import json
import math
import os
import pickle
import random
from collections import defaultdict
import dgl
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import BertModel

from models.GDGN import Bert
from utils import get_cuda,get_all_entity
entity_type2id = {"OTH":0,'LOC': 1, 'TIME': 2, 'MISC': 3, 'PER': 4, 'ORG': 5, 'NUM': 6} 
IGNORE_INDEX = -100

# 继承data 中的IterableDataset 
# 这个类是给Glove 算法加载数据的
class DGLREDataset(IterableDataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):

        super(DGLREDataset, self).__init__()

        # record training set mention triples
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        self.INTRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
                self.instance_in_train = info['intrain_set']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            print('loading..')
            self.data = []

            for i, doc in enumerate(ori_data):

                title, entity_list, labels, sentences = \
                    doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents']

                Ls = [0]
                L = 0
                for x in sentences:
                    L += len(x)
                    Ls.append(L)
                for j in range(len(entity_list)):
                    for k in range(len(entity_list[j])):
                        sent_id = int(entity_list[j][k]['sent_id'])
                        entity_list[j][k]['sent_id'] = sent_id

                        dl = Ls[sent_id]
                        pos0, pos1 = entity_list[j][k]['pos']
                        entity_list[j][k]['global_pos'] = (pos0 + dl, pos1 + dl)

                # generate positive examples
                train_triple = []
                new_labels = []
                for label in labels:
                    head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence']
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)
                    label['r'] = rel2id[relation]

                    train_triple.append((head, tail))

                    label['in_train'] = False

                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else:
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break

                    new_labels.append(label)

                # generate negative examples
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:
                            na_triple.append((j, k))

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)
                if len(words) > self.document_max_length:
                    words = words[:self.document_max_length]

                word_id = np.zeros((self.document_max_length,), dtype=np.int32)
                pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
                ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
                mention_id = np.zeros((self.document_max_length,), dtype=np.int32)

                for iii, w in enumerate(words):
                    word = word2id.get(w.lower(), word2id['UNK'])
                    word_id[iii] = word

                entity2mention = defaultdict(list)
                mention_idx = 1
                already_exist = set()  # dealing with NER overlapping problem
                for idx, vertex in enumerate(entity_list, 1):
                    for v in vertex:
                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']
                        if (pos0, pos1) in already_exist:
                            continue
                        pos_id[pos0:pos1] = idx
                        ner_id[pos0:pos1] = ner2id[ner_type]
                        mention_id[pos0:pos1] = mention_idx
                        entity2mention[idx].append(mention_idx)
                        mention_idx += 1
                        already_exist.add((pos0, pos1))
                
                assert pos_id.max() == len(entity_list)
                
                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]                                
                self.data.append({
                    'title': title,
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id': pos_id,
                    'ner_id': ner_id,
                    'mention_id': mention_id,
                    'entity2mention': entity2mention,
                    'overlap': new_overlap
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

        if opt.k_fold != "none":
            k_fold = opt.k_fold.split(',')
            k, total = float(k_fold[0]), float(k_fold[1])
            a = (k - 1) / total * len(self.data)
            b = k / total * len(self.data)
            self.data = self.data[:a] + self.data[b:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


class BERTDGLREDataset(IterableDataset):
    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):
        super(BERTDGLREDataset, self).__init__()
        # record training set mention triples 
        # set([]) 和 set() 是同样的作用
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        
        # 如下这几个参数的含义与作用？
        self.INFRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr) # info <class 'dict'> key-value 是：data, intrain_set
                # list, 里面是dict                       
                self.data = info['data']
                # ('Justin Broadrick', 'Earache Records', 'P264')
                self.instance_in_train = info['intrain_set'] # 是一个三元组组成的list
            print('load preprocessed data from {}.'.format(save_file))
            
        else: # 在没有缓存数据时（.pkl文件不存在）进行的操作。 数据处理操作需要认真阅读！
            bert = Bert(BertModel, 'bert-base-uncased', opt.bert_path)
            entity2id = get_all_entity(path="../data/all.json") # 获取entity->id
            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            print('loading..')
            self.data = []

            for i, doc in enumerate(ori_data):
                # doc 中的内容只有如下四项，分别是 title, vertexSet, labels, sents
                title, entity_list, labels, sentences = \
                    doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents']
                cur_entity_id = [] # 全局的 entity_id，使用 get_all_entity() 获取
                local_entity_id = [] # 局部的 entity_id，每篇doc中 的entity id 从0开始计数
                for mentions in entity_list:
                    name = mentions[0]['name'] # 仅根据第一个mention的name 来获取
                    cur_entity_id.append(entity2id[name])

                Ls = [0] # Ls[i] 表示的就是第i条句子开始的绝对长度。直接以原始setence中的word为基准
                L = 0
                # step1. 遍历每个的 sentence 的长度，累计得到当前的总长度L， 并将其放到 Ls中。
                for x in sentences: 
                    L += len(x)
                    Ls.append(L)
                
                # step2. 遍历每个entity
                for j in range(len(entity_list)): 
                    for k in range(len(entity_list[j])): # 找出当前entity下的mention个数
                                                
                        sent_id = int(entity_list[j][k]['sent_id']) # 找出当前这个mention 所在的sent_id ，但是如果一个mention
                        # 下面这行代码应该无用
                        # entity_list[j][k]['sent_id'] = sent_id # 即使相同的mention，但是可能出现在句中的位置不同，所以这里放的是[实体_id][mention_id]['send_id'] = sent_id

                        dl = Ls[sent_id]  # dl 是对应第 sent_id 个句子的长度
                        mention_start, mention_end = entity_list[j][k]['pos'] # 得到这个mention 的 [start,end] 坐标
                        entity_list[j][k]['global_pos'] = (mention_start + dl, mention_end + dl) # global_pos 计算的就是全局位置
                
                # step3. 遍历labels
                # generate positive examples
                train_triple = []
                new_labels = []  # 产生新的label标签
                for label in labels:
                    head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence'] # 从原始数据中取出对应的字段
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation) # 判断是否有对应的relation
                    label['r'] = rel2id[relation] # 将rel => id
                    train_triple.append((head, tail)) # 记录正样本数据
                    label['in_train'] = False  # 为什么这里初始化为False?
                    
                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)  # 这相当于根据所有的mention都生成一个train example  => 这会生成很多个训练数据
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else: # 因为这里存在既加载train，又加载dev/test。所以为了判断是否train中的标签会出现dev中，所以就用来这个标记。
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break
                                       
                    new_labels.append(label)

                # generate negative examples  => 其实负样本的选择也是很重要的，所以这里也是一个课题可以作为改进
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:  # 如果不在正样本数据中，则作为负样本使用
                            na_triple.append((j, k))

                # generate document ids
                words = [] # 取出sentence 中的所有单词
                # a = []
                for sentence in sentences:
                    words.extend(sentence)
                    # for word in sentence:
                    #     a.append(word)
                # print(a)
                # print(words)
                # bert_token, bert_starts, bert_subwords 用bert_xx 是为了表示要送给bert处理
                # bert_starts[i] 表示第i个word 在bert_token 中的起始下标。因为有拆分情况，所以需要记录一下
                bert_token, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(words)
                # zeros() 函数会传入一个shape，即下面的(self.document_max_length,) 这里将传递给杜设置成document_max_length，就可以猜测是要对整个tokenizer之后的序列进行一个标注，否则没有必要搞这么个长度
                word_id = np.zeros((self.document_max_length,), dtype=np.int32) 
                pos2entityid = np.zeros((self.document_max_length,), dtype=np.int32)  # 这个改做pos2entityid 比较合理
                ner_id = np.zeros((self.document_max_length,), dtype=np.int32) # 默认的初始值就代表这个不是NER
                
                # 这个改做叫pos2mention_id 更好
                mention_id = np.zeros((self.document_max_length,), dtype=np.int32)
                word_id[:] = bert_token[0] # 其实就是 tokenizer处理后的 input_ids

                entity2mention = defaultdict(list) # 记住每个entity 对应mention的id
                mention_idx = 1 # 对mention的下标计数，注意是从1开始。 即统计一篇 doc 中的所有mention个数，所以是个全局变量
                already_exist = set() # 记录两个
                for idx, vertex in enumerate(entity_list, 1): # 对所有的entity遍历
                    for v in vertex: # vertex 代表的是这个entity的所有 mention 集合
                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']  # 注意这里取得是绝对位置
                        pos0 = bert_starts[pos0] # 返回在tokenizer之后（原words在）位置pos0处的位置
                        pos1 = bert_starts[pos1] if pos1 < len(bert_starts) else 1024 # TODO 这个值为啥选择1024？
                        # 下面这两种if会存在吗？
                        if (pos0, pos1) in already_exist:
                            continue

                        if pos0 >= len(pos2entityid):
                            continue
                        
                        pos2entityid[pos0:pos1] = idx # 表示pos0 -> pos1 这个位置是第idx个实体（用的是idx下标）
                        ner_id[pos0:pos1] = ner2id[ner_type]  # 标记整个位置的 ner_type 情况，标记成id。 ner_id[i] 表示成第i个位置是个ner，且这个ner的类型对应的id是 ner2id[ner_type]
                        mention_id[pos0:pos1] = mention_idx # 记录从pos:pos1 这个位置是一个mention（用的是mention_idx下标），对应第mention_idx个mention
                        entity2mention[idx].append(mention_idx)  
                        mention_idx += 1  
                        already_exist.add((pos0, pos1))
                replace_i = 0
                # TODO 功能存疑
                idx = len(entity_list) # 这行代码放在这个位置，应该才ok 
                if entity2mention[idx] == []:                    
                    entity2mention[idx].append(mention_idx)
                    while mention_id[replace_i] != 0:
                        replace_i += 1
                    mention_id[replace_i] = mention_idx
                    pos2entityid[replace_i] = idx
                    ner_id[replace_i] = ner2id[vertex[0]['type']]
                    mention_idx += 1

                new_Ls = [0] # 之前的Ls是按照原始sentence的分词搞的，接下来就要用基于bert tokenizer 之后的来搞
                for ii in range(1, len(Ls)):
                    # bert_starts[Ls[ii]] 找出对应 Ls[ii] 这个word 得到的token下标
                    new_Ls.append(bert_starts[Ls[ii]] if Ls[ii] < len(bert_starts) else len(bert_subwords))
                Ls = new_Ls

                
                assert pos2entityid.max() == len(entity_list)
                
                # 没有明白这个 overlap 的作用是什么？
                # 这个应该是作者和别人模型（LSR）比较时需要使用到的一个参数
                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]                
                self.data.append(
                    {
                    'title': title,
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id': pos2entityid,
                    'ner_id': ner_id,
                    'mention_id': mention_id,
                    'entity2mention': entity2mention,                    
                    'overlap': new_overlap,
                    'entity_id': cur_entity_id,  # 获取的是全局的entity_id，用于建全局图使用
                    'local_entity_id': local_entity_id  # 当前doc的entity_id，按照句子的顺序获取
                    }
                )
            
            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


class DGLREDataloader(DataLoader):
    # 使用num_workers 加速数据的准备过程
    # 这个 h_t_limit_per_batch 是什么意思？
    # TODO h_t_limit 又是啥？=> 我猜测是 h_entity 和 tail_entity 的连接数。这里的1722 是因为 42*42-42 = 1722
    def __init__(self, dataset, batch_size, shuffle=False, h_t_limit_per_batch=300, h_t_limit=1722, relation_num=97,max_length=512, negativa_alpha=0.0, dataset_type='train'):
        super(DGLREDataloader, self).__init__(dataset, batch_size=batch_size) # 初始化 DataLoader 用多线程
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.negativa_alpha = negativa_alpha
        self.dataset_type = dataset_type

        self.h_t_limit_per_batch = h_t_limit_per_batch
        self.h_t_limit = h_t_limit
        self.relation_num = relation_num
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.order = list(range(self.length))        
        self.entity2id = get_all_entity(path="../data/all.json")

    def __iter__(self):
        # shuffle
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)  # 计算按照当前的batch_size设置，会有多少个batch？
        # 根据当前的idx，返回每个batch需要的数据
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)] # 取出batch对应 的 order
        
        # 查看每个小 batch 里面
        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch) # 因为每个batch 不一定是整除的，所以这里先计算一下当前batch 中的个数
            
            context_word_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_pos_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_ner_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_mention_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_word_mask = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_word_length = torch.zeros(self.batch_size,dtype=torch.long).cpu()
            ht_pairs = torch.zeros(cur_bsz, self.h_t_limit, 2,dtype=torch.long).cpu() # h_entity 和 tail_entity 组成的对就是 ht_pairs
            # 这里生成的大小就是 [cur_bsz,self.h_t_limit,2]。 并且是以0填充
            # 与上面不同，这里h_entity 和 t_entity 使用的全局id索引
            ht_pairs_global = torch.zeros(cur_bsz, self.h_t_limit, 2,dtype=torch.long).cpu() 
            ht_pair_distance = torch.zeros(self.batch_size, self.h_t_limit,dtype=torch.long).cpu()
            ht_pairs_entity_type_id = torch.zeros(cur_bsz, self.h_t_limit,2, dtype=torch.long).cpu() # 初始值为0

            # 这个变量的含义是什么？
            # realtion_multi_label[i][j][k]=1表示 第i篇doc中的第j个训练样本具有k这种关系；
            # realtion_multi_label[i][j][k]=0表示 第i篇doc中的第j个训练样本不具有k这种关系；
            # 第i篇doc中的第j个训练样本是哪两个实体组成可以通过 ht_pairs 来找到
            relation_multi_label = torch.zeros(self.batch_size, self.h_t_limit, self.relation_num).cpu() # zeros 得到的结果是float
            relation_label = torch.zeros(self.batch_size, self.h_t_limit,dtype=torch.long).cpu()
            # self.h_t_limit 是因为
            # 不理解这里的relation_mask 的作用 => 因为同一batch中有的doc没有那么多的relation label，所以这里有个mask操作。
            relation_mask = torch.zeros(self.batch_size, self.h_t_limit,dtype=torch.long).cpu() 
            relation_label.fill_(IGNORE_INDEX)
            
            # 记录 type * type * relation 下的label情况。因为是实体类型，所以就没有了 doc 那个维度，而是直接把整个batch的信息合在一起
            # 把同一个 batch 下的所有数据统计出来
            type_relation_label = torch.zeros(7*7+1, self.relation_num).cpu()
            idx_cnt = defaultdict(int) # 用于记录每个type*type 得到的index 的累积值
            max_h_t_cnt = 0

            label_list = []
            L_vertex = []
            titles = []
            indexes = [] # TODO: ?            
            entity2mention_table = [] # 这个是什么？ => 每篇 doc 形成的一个矩阵（entity 和 mention的对应关系）叫做entity2mention。将batch下的每篇doc得到的矩阵放到同一个list中
            path_table = []
            batch_entity_id = []
            overlaps = []
            doc_hentity_tentity2_idx = {} # doc +h_idx + t_idx => pair id
            # 对这个batch 中的数据进行处理
            for i, example in enumerate(minibatch):
                title, entities, labels, na_triple, word_id, pos_id, ner_id, mention_id, entity2mention, entity_id = \
                    example['title'], example['entities'], example['labels'], example['na_triple'], \
                    example['word_id'], example['pos_id'], example['ner_id'], example['mention_id'], example[
                        'entity2mention'], example['entity_id']
                
                overlaps.append(example['overlap'])
                # entity2mention_t 是个矩阵的形式，为啥要多加1？
                # 因为第一个是留给cls,所以+1
                entity2mention_t = get_cuda(torch.zeros((pos_id.max() + 1, mention_id.max() + 1)))
                # entity2mention 是个dict。 这两个for循环是为了构建一个邻接矩阵，将entity2mention 转换成一个邻接矩阵
                for e, ms in entity2mention.items():
                    for m in ms:
                        entity2mention_t[e, m] = 1
                entity2mention_table.append(entity2mention_t)

                L = len(entities)  # 当前这篇doc 中entity 的数量
                word_num = word_id.shape[0]
                # 将word_id 的值放到context_word_id[i,:word_num] 中。 这里的 :word_num 是有点儿多余
                context_word_ids[i, :word_num].copy_(torch.from_numpy(word_id))
                context_pos_ids[i, :word_num].copy_(torch.from_numpy(pos_id))
                context_ner_ids[i, :word_num].copy_(torch.from_numpy(ner_id))
                context_mention_ids[i, :word_num].copy_(torch.from_numpy(mention_id))

                idx2label = defaultdict(list) # 标签数据的转化
                label_set = {}
                for label in labels:
                    head, tail, relation, intrain, evidence = \
                        label['h'], label['t'], label['r'], label['in_train'], label['evidence']
                    idx2label[(head, tail)].append(relation)
                    label_set[(head, tail, relation)] = intrain # 这个intrain 是判断当前这条样例是否出现在train中的标志

                label_list.append(label_set)
                # idx2label 的值如下：{ (2, 3): [1], (11, 3): [3], (0, 1): [7] ... }
                if self.dataset_type == 'train':                    
                    train_triple = list(idx2label.keys()) # 得到h_entity, t_entity 
                    # 之前的 ht_pairs 存放的是句子内的idx，所以使用一个映射关系找出对应的全局id
                    for j, (h_idx, t_idx) in enumerate(train_triple):
                        doc_hentity_tentity2_idx[str(i)+"_"+str(h_idx)+"_"+str(t_idx)] = j
                        hlist, tlist = entities[h_idx], entities[t_idx] # 得到h_idx,t_idx 对应的mention list
                        h_entity_name = hlist[0]['name'] #使用第一个名字
                        t_entity_name = tlist[0]['name']
                        
                        # 得到头尾实体的类型
                        h_entity_type_id = entity_type2id[hlist[0]['type']]
                        t_entity_type_id = entity_type2id[tlist[0]['type']]
                        cur_idx = (h_entity_type_id-1)*7+t_entity_type_id # 计算出标签位置
                        idx_cnt[cur_idx] += 1

                        global_h_entity_idx = self.entity2id[h_entity_name]
                        global_t_entity_idx = self.entity2id[t_entity_name]
                        
                        # 这里global_h_entity_idx+1 的原理和下面 h_idx+1 的原因是一样的
                        # 在模型的forward()函数里面，就需要针对这个ht_pairs_global 进行一个 entity embedding 的选择
                        ht_pairs_global[i,j,:] = torch.Tensor([global_h_entity_idx+1, global_t_entity_idx+1])
                        # if global_h_entity_idx == 0:  # global_h_entity_idx is 0 是允许的，因为字典下标是从0计数的
                        #     print("bug is here")
                        
                        # 这里加一，是因为什么？原因可能有两个：
                        # （1）因为可能有个root节点的编号是0
                        # （2）上面在初始化的时候用0初始化，但是的的确确会有 h_idx 与 t_idx 为0，所以这里+1 【我认为是这个原因】
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1]) # i表示第i篇doc（从0开始计数）， j表示第j个样本（从0计数）。后面在针对负样本进行生成的时候，会接着使用这个j

                        # 实体对中实体的类型信息，这里直接转换成一个数
                        # ht_pairs_entity_type_id[i, j] = torch.Tensor([(h_entity_type_id-1)*7+t_entity_type_id])
                        ht_pairs_entity_type_id[i, j,:] = torch.Tensor([h_entity_type_id,t_entity_type_id])

                        label = idx2label[(h_idx, t_idx)]
                        # TODO: 为什么下面还要计算 ht_pair_distance ？
                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2

                        for r in label: # 可能一对实体有多个 relation，所以这里有个for循环
                            relation_multi_label[i, j, r] = 1
                            type_relation_label[cur_idx,r] += 1 # 因为可能不止一个标签，所以要增加1

                        relation_mask[i, j] = 1
                        rt = np.random.randint(len(label)) # 从中label list中随机取一个作为rt，然后作为relation_label[i,j] 
                        # 每行代表一个(h_entiyt,t_entity)之间的关系标签 => 错！！！ 这么理解是错误的。
                        # relation_label[i][j] 表示的： 第i篇doc，第j个样本具有的关系是 label[rt]
                        # Bug 但是这里存在一个问题。 我们对多种关系其实都需要预测的，这样的话只能预测某一种了
                        relation_label[i, j] = label[rt] 
                    # 加入负样本进行学习
                    lower_bound = len(na_triple) # lower_bound 代表的是负样本的个数
                    if self.negativa_alpha > 0.0: # self.negativa_alpha = 4.0
                        random.shuffle(na_triple) # 随机 shuffle
                        lower_bound = int(max(20, len(train_triple) * self.negativa_alpha)) # 确定负样本个数的下界
                    # 重复上面这个逻辑，只不过这里是针对负样本的。注意这里 len(train_triple) 的值会被初始化为j
                    for j, (h_idx, t_idx) in enumerate(na_triple[:lower_bound], len(train_triple)):
                        doc_hentity_tentity2_idx[str(i)+"_"+str(h_idx)+"_"+str(t_idx)] = j
                        hlist, tlist = entities[h_idx], entities[t_idx]
                        # 可以整体赋值，这里就是赋值一个向量。
                        # 这里存储的是实体的相对位置
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1])  # 注意这里的 ht_pairs 和 使用的relation_multi_label 是有对应关系的。 对应关系正如前定义relation_multi_raltion时所述
                        h_entity_name = hlist[0]['name']
                        t_entity_name = tlist[0]['name']
                        h_entity_type_id = entity_type2id[hlist[0]['type']]
                        t_entity_type_id = entity_type2id[tlist[0]['type']]
                        cur_idx = (h_entity_type_id-1)*7+t_entity_type_id # 计算出标签位置
                        global_h_entity_idx = self.entity2id[h_entity_name]
                        global_t_entity_idx = self.entity2id[t_entity_name]
                        
                        # 存储实体的绝对位置
                        ht_pairs_global[i,j,:] = torch.Tensor([global_h_entity_idx+1, global_t_entity_idx+1])
                        # ht_pairs_entity_type_id[i, j] = torch.Tensor([(h_entity_type_id-1)*7+t_entity_type_id])
                        ht_pairs_entity_type_id[i, j,:] = torch.Tensor([h_entity_type_id,t_entity_type_id])

                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2
                        # 因为是负样本，所以这里的 r=0
                        relation_multi_label[i, j, 0] = 1
                        relation_label[i, j] = 0
                        relation_mask[i, j] = 1
                        type_relation_label[cur_idx,0] += 1
                        idx_cnt[cur_idx] += 1
                        max_h_t_cnt = max(max_h_t_cnt, len(train_triple) + lower_bound)
                                        
                else:
                    j = 0 
                    for h_idx in range(L): # 因为要判断所有的entity，所以用的是双重for循环
                        for t_idx in range(L):
                            if h_idx != t_idx: # 这里判断是否赋值 ht_pairs 的条件只是 h_idx 和 t_idx 的值的比较
                                doc_hentity_tentity2_idx[str(i)+"_"+str(h_idx)+"_"+str(t_idx)] = j
                                hlist, tlist = entities[h_idx], entities[t_idx] # 分别获取每个实体下的所有mention 信息
                                # TODO ht_pairs 是什么意思？  +1 是因为？ 难道就只是想看h_entity 和 tail_entity 之间能否组成一对？所以就叫ht_pair? => 见上分析
                                ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1]) 
                                relation_mask[i, j] = 1
                                
                                
                                h_entity_name = hlist[0]['name']
                                t_entity_name = tlist[0]['name']
                                global_h_entity_idx = self.entity2id[h_entity_name]
                                global_t_entity_idx = self.entity2id[t_entity_name]
                                
                                # 存储实体的绝对位置
                                ht_pairs_global[i,j,:] = torch.Tensor([global_h_entity_idx+1, global_t_entity_idx+1])

                                
                                # 这里只取hlist[0][x][x] 中的[0] 是为何？
                                delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                                if delta_dis < 0:
                                    ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                                else:
                                    ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2
                                
                                j += 1 # 只有组成一对之后，j++
                    # 找出最大的一个 h_t_cnt， 这个 max_h_t_cnt 的最大值是930=31*31-31
                    max_h_t_cnt = max(max_h_t_cnt, j)
                    L_vertex.append(L)
                    titles.append(title)
                    indexes.append(self.batches_order[idx][i])
                # end else
                batch_entity_id.append(entity_id)
            context_word_mask = context_word_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()
            
            # 对 type_relation_label 做一个归一化处理                    
            for item in idx_cnt.items():
                key,val = item
                type_relation_label[key] /= val
                # print("-----:",key,max(type_relation_label[key]),val)

            yield {'context_idxs': get_cuda(context_word_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_pos': get_cuda(context_pos_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_ner': get_cuda(context_ner_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_mention': get_cuda(context_mention_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_mask': get_cuda(context_word_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_length': get_cuda(context_word_length[:cur_bsz].contiguous()),
                   'h_t_pairs': get_cuda(ht_pairs[:cur_bsz, :max_h_t_cnt, :2]),
                   'relation_label': get_cuda(relation_label[:cur_bsz, :max_h_t_cnt]).contiguous(),
                   'relation_multi_label': get_cuda(relation_multi_label[:cur_bsz, :max_h_t_cnt]),
                   'relation_mask': get_cuda(relation_mask[:cur_bsz, :max_h_t_cnt]),                   
                   'labels': label_list,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'indexes': indexes,                   
                   'entity2mention_table': entity2mention_table,                   
                   'overlaps': overlaps,
                   'batch_entity_id': batch_entity_id,
                   'h_t_pairs_global':get_cuda(ht_pairs_global[:cur_bsz, :max_h_t_cnt, :2]),
                   'h_t_pairs_entity_type_id': get_cuda(ht_pairs_entity_type_id[:cur_bsz, :max_h_t_cnt,:]),# 生成每个实体对的类型信息
                   'type_relation_label':get_cuda(type_relation_label), # 生成type_relation 之间的label 信息
                   'doc_hentity_tentity2_idx':doc_hentity_tentity2_idx # doc+h_entity_idx+t_entity_idx 得到实体对的下标
                   }