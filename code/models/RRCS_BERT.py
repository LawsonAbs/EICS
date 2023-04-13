import datetime
'''
2021/12/25
使用 图+RGCN 的方法，动态地建立 realtion entity graph
（1）使用图
（2）使用R-GCN 推导关系
（3）使用外部KB解决常识获取问题

2022/01/10
（1）生成的图不再batch，分开操作 => 速度会慢一些
'''
import random
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer,BertConfig
from utils import get_cuda,get_labelid2name



class RRCS_BERT(nn.Module):
    ''' Parameters
        config ：
    '''
    def __init__(self, config):
        super(RRCS_BERT, self).__init__()
        self.config = config
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        # 下面这两个都是在需要用到的时候，才使用这个去得到具体的数
        if config.use_entity_type:
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)
        # 这个 use_entity_id 值得思考，到底是怎么来的？
        if config.use_entity_id:
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)
        
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.gcn_dim = config.gcn_dim
        # 是否固定bert？ 即不更新bert的参数
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False
        
        self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)
        
        
        # 过完Bert的预测，得到各个entity pair之间的 multi_relation。
        self.first_predict = nn.Sequential(            
            nn.Linear(self.gcn_dim * 2, self.gcn_dim ),
            self.activation,  # TODO: 这个地方的激活函数没有必要用 relu
            self.dropout,
            nn.Linear(self.gcn_dim , config.relation_num),            
        )

        self.relid2name = get_labelid2name()
    def forward(self, **params): # 对一个batch 进行操作
        '''
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]        
        entity2mention_table: list of [local_entity_num, local_mention_num]        
        h_t_pairs: [batch_size, h_t_limit, 2]
        '''
        input_ids = params['words'] # (batch_size,max_length) 这个input_ids 就是针对输入数据得到的input_ids
        mask = params['mask'] 
        
        # h_t_type = params['h_t_type'] # 得到当前batch的实体类型 => TODO: 这个要在data 中获取
        max_entity_num = torch.max(params['entity_id']) # 获取当前批次所有doc中最大的实体数
        batch_entity_num = torch.max(params['entity_id'],dim=1)[0] # 获取idx
        bsz, slen = input_ids.size()
        h_t_type = torch.zeros(bsz,max_entity_num)
        # =========================== step 1. 获取原始句子、mention、entity 的表示  ===========================        
        # 先过bert.  encoder_outputs 是 last_hidden_states  
        output = self.bert(input_ids=input_ids, attention_mask=mask)
        encoder_outputs = output.last_hidden_state
        sentence_cls = output.last_hidden_state[:,0,:]
        
        # if self.config.use_entity_type:
        encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type']),self.entity_id_emb(params['entity_id'])], dim=-1)

        # if self.config.use_entity_id:
        # encoder_outputs = torch.cat([encoder_outputs, ], dim=-1)
        # 在768后面再追加指定的维度，为什么要追加这个表示？？？ => 因为想保留实体的类型和id
        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),
            dim=-1)
        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]
        # sentence_cls: [batch_size, bert_hid+type_size+id_size]               


        # =========================== step 2. 获取各个doc中的 mention 表示 => 进而得到entity的表示  ===========================
        # tokenizer 之后得到的token，在每个位置是否属于mention 的下标
        # 做什么用的？ => 一个batch中的所有doc的所有 mention 拼接在一起得到的表示，这个mention的表示是按照
        # 每个mention所占的 token 加权平均得到的    
        mention_id = params['mention_id']
        # mention -> entity
        # entity2mention_table 这个是一篇doc中 entity 和 mention 形成的矩阵(其大小是entity_num * mention_num )。矩阵中的值是怎么样形成的？  => 通过数据分析可以获取
        # eneity2mention[i][j]=1 表示的就是 第i个entity和 第j个mention 是同一个实体，相反，如果没有联系，则为0
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
                
        # 获取一个batch中所有篇章中所有 entity 的表示
        batch_entity_feateature = torch.zeros(bsz,max_entity_num,808,dtype=torch.float32).cuda()
        # features size = [all_mention_num,808] 这里all_mention_num 是一个batch中所有的mention 数目

        # 上面的 batch_entity_features 的size固定，
        # 这里的 batch_feature_bert 是list，没有固定的size，追加的顺序是和entity id的顺序相同，所以后面可以直接用这个给entity_graph 的图作输入
        batch_feature_bert = []
        for i in range(bsz): # batch是多少，就有几个图。
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]  拿到每篇 doc 中每个token的 embedding 表示
            mention_num = torch.max(mention_id[i]) # 这个参数的含义是：这个doc中 vertexSet中mention 的个数（一篇doc所有mention 的个数）num(mention) >= num(entity)
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen] select_metrix[i][j] =1代表的就是第i个mention在位置j出现过，否则没有出现。select_metrix.size(1) 就是doc 的长度 
            # average word -> mention  找出每行有几个数。 相当于一个 average pool
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
            # 根据select_metrix 和 encoder_output 做乘法，得到各个 mention 的表示
            x = torch.mm(select_metrix, encoder_output)  # [mention_num, bert_hid]
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0) # 拼接整个句子的表示            

            # average mention -> entity  # 这个 entity2mention_tabel 本身就是float类型
            select_metrix = entity2mention_table[i].float()  # [local_entity_num, mention_num]  对每个图（doc）进行操作， cur_entity2mention_table[i] 表示的就是第i张图（doc）
            select_metrix[0][0] = 1 
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1)) # 做这个扩展操作是为了后面的除法
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)  # 这个操作同上面，也是为了保持每行的值相同 。这里就是为了计算每个下标该有多大的权重去获取最后的entity表示。相当于对 mention 的一个加权
            entity_representation = torch.mm(select_metrix, x) # 通过合并mention的特征就能获取到 entity_representation
            # 同理，这个select_metrix.size(0) 就是entity_num 。所以这里就相当于给一篇doc中的所有的 entity 赋特征值。 将所有的entity 属性表示放到entity_bank 中
            batch_entity_feateature[i, :select_metrix.size(0) - 1] = entity_representation[1:]
            batch_feature_bert.append(entity_representation[1:])
        
        # =========================== step 3.  ===========================                
        # 选择当前batch 中最大数目的entity 作为 entity_num （在dev_44.json 中，其值为31）
        # 当前定义的变量都是为了后面的程序服务的！！因为后面要用到 entity_num，即需要知道当前批次最大的个数是什么
        max_entity_num = torch.max(params['entity_id'])

        # global_info 是干什么的？ => 获取整个文档的信息，这个会放在最后面作为一个特征用于预测
        global_info = torch.zeros(bsz, self.bank_size).cuda()

        # =========================== step 4. 获取各个 entity pair的表示，判断entity pair之间的多重关系 ===========================
        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)  # 这个参数是什么意思？ => 见上分析

        # 找出 h_entity/t_entity 的index 过程 
        # [batch_size, h_t_limit, bank_size] 为什么要搞成这个形状 => 因为后面要做一个gather操作，需要保证每行的值都是相同的
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, 808)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, 808)
        # [batch_size, h_t_limit, bank_size]

        # 从batch_entity_feat 中得到每个 entity 的表示，称作是h_entity/t_entity
        h_entity = torch.gather(input=batch_entity_feateature, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=batch_entity_feateature, dim=1, index=t_entity_index)
        # 根据每句话中的 entity pair 来判断关系，得到的是局部的 prediction
        # 根据这预测得到的句内 entity之间的 prediction，然后根据这个 prediction 建图
        # [bsz,entity_pair_num,relation_num]
        entity_pair_multi_relation_predict = self.first_predict(torch.cat((h_entity,t_entity),dim=-1))                
        
        return  entity_pair_multi_relation_predict