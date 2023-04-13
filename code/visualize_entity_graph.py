import matplotlib.pyplot as plt
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
from transformers import BertModel
from models.GDGN import Bert
plt.figure(figsize=(8, 8))
IGNORE_INDEX = -100


class BERTDGLREDataset():
    def __init__(self, src_file, ner2id, rel2id,dataset_type='train'):
        super(BERTDGLREDataset, self).__init__()
        # record training set mention triples                
        self.data = None
        self.document_max_length = 512
        
        # 如下这几个参数的含义与作用？
        self.INFRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        
        bert = Bert(BertModel, 'bert-base-uncased', "/storage/lawson/pretrain/bert-base-uncased")
        with open(file=src_file, mode='r', encoding='utf-8') as fr:
            ori_data = json.load(fr)
        
        self.data = []

        for i, doc in enumerate(ori_data):
            # doc 中的内容只有如下四项，分别是 title, vertexSet, labels, sents
            title, entity_list, labels, sentences = \
                doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents']
            
            # 节点命名数据
            node_data  = [] 

            Ls = [0] # Ls[i] 表示的就是第i条句子开始的绝对长度。直接以原始setence中的word为基准
            L = 0
            # step1. 遍历每个的 sentence 的长度，累计得到当前的总长度L， 并将其放到 Ls中。
            for x in sentences: 
                L += len(x)
                Ls.append(L)
            
            # step2. 遍历每个entity
            for j in range(len(entity_list)): 
                node_data.append(entity_list[j][0]['name'])
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
                new_labels.append(label)            

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

            # 画出节点的名称

            node_labels = { index:data for index,data in enumerate(node_data) }
            # construct entity graph & path
            # 这是单个doc中的entity_graph
            path = "../data/entity_graph_all/dev_"+str(i)+".png"
            a,b = self.create_entity_graph(Ls, pos2entityid)            

            connect_edges = [(i,j) for i,j in zip(a,b)] # 得到连接边
            # 独自使用nx建图
            G = nx.Graph() # 新建一个图
            # 因为有离散点，所以这里逐个添加所有点
            for i in range(pos2entityid.max()):
                G.add_node(i) # 逐个添加节点
                
            for left,right in zip(a,b):
                G.add_edge(left,right,color='black',weight=2)

            # 加入golden label，边的颜色不同
            for i,j in train_triple:
                if (i,j) in connect_edges:
                    G.add_edge(i,j,color='blue',weight=20) # 如果二者边重合，则显示为蓝色
                else:
                    G.add_edge(i,j,color='red',weight=20) # 如果仅有golden label 边，则显示为红色
            edges = G.edges()
            colors = [G[u][v]['color'] for u,v in edges]

            pos = nx.circular_layout(G)    # 图像使用圆形布局                    
            pos_higher = {}

            for k, v in pos.items():  #调整下顶点属性显示的位置，不要跟顶点的序号重复了
                if(v[1]>0):
                    pos_higher[k] = (v[0]-0.04, v[1]+0.04)
                else:
                    pos_higher[k] = (v[0]-0.04, v[1]-0.04)

            nx.draw(G, pos,edge_color=colors, node_size=300, with_labels=True) # 画图，设置节点大小
            nx.draw_networkx_labels(G,pos_higher, labels=node_labels,font_color="brown", font_size=10)  # 将desc属性，显示在节点上        
            plt.savefig(path,dpi=600,format='png') # 保存矢量图
            plt.close() # 防止图片重叠
     
    # Entity level 的图之间的边是怎么构建的？ =>  合并所有实体间的边，这些边连接相同的两个mention
    # TODO 这里的建边过程是否还可以值得优化一下？
    # Ls[i] 表示的就是第i条句子开始的绝对长度（在tokenizer之后的）
    def create_entity_graph(self, Ls, pos2entity_id):
        # 新建一个空图 Class for storing graph structure and node/edge feature data.
        # 但是这个图好像是同构图
        # graph = dgl.DGLGraph()
        # graph 中的节点编号是 [0,postion2entity_id.max()-1]，所以相当于加入了 position2entity_id.max() 个节点到图中。
        # 有多少个实体，节点就是多少。 先把节点数表示出来，但是这些节点的具体特征留到后面再做处理
        # add_nodes() 函数是给图中添加节点。后面会有添加边的操作。
        # graph.add_nodes(pos2entity_id.max())
        d = defaultdict(set)

        for i in range(1, len(Ls)): # 找出所有句子的长度
            tmp = set() # 判断当前 sentence 中有几个实体，并使用set记录它们的id
            for j in range(Ls[i - 1], Ls[i]): # 获取每条 sentence 中的实体
                if pos2entity_id[j] != 0:
                    tmp.add(pos2entity_id[j])
            tmp = list(tmp)  # [1,2,3,4,5]
            for ii in range(len(tmp)): # 使用双重 for 循环在同一个sentence之间的实体间建立关系（边）
                for jj in range(ii + 1, len(tmp)):
                    d[tmp[ii] - 1].add(tmp[jj] -1 ) # 建双向边
                    d[tmp[jj] - 1].add(tmp[ii] -1) # 因为图中的节点编号是从0开始，所以这里有个减一操作

            # 加入doc节点
            # for i in tmp:
            #     d[0].add(i)
            #     d[i].add(0)

        # 将d拆分，形成一一对应的数组，然后交由图进行创建边的操作
        a = [] 
        b = []
        for k, v in d.items():
            for vv in v:
                a.append(k)
                b.append(vv)
        
        return (a, b) # 返回边信息
        # graph.add_edge(a,b)
        # return graph


data_dir = '../data/'
rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), "r"))
id2rel = {v: k for k, v in rel2id.items()}
word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), "r"))
ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), "r"))
train_set = BERTDGLREDataset('../data/dev_44.json', ner2id, rel2id,dataset_type='train')