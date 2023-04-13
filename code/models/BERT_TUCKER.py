import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import GraphConv
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from transformers import BertModel,BertTokenizer

from utils import get_cuda,get_pair_nodes



class TuckER(torch.nn.Module):
    def __init__(self, relation_num, d1, d2, **kwargs):
        super(TuckER, self).__init__()
        # self.E = torch.nn.Embedding(entity_num, d1) # 实体的Embedding 
        self.R = torch.nn.Embedding(relation_num, d2) # 关系的Embedding
        
        # 核心张量，三维
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d2)
        

    def init(self):
        # xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
    
    # head_entity 表示的是头实体的向量
    # tail_entity 表示的则尾实体的向量

    def forward(self, head_entity, tail_entity):
        bsz = head_entity.size(0) # 得到当前的实体个数

        # step1. 获取输入的表示
        # head_entity = self.E(head_entity_idx) # 维度是[n,d1]，其中n是实体的个数
        # x = self.bn0(head_entity) # 归一化
        #   
        x = self.input_dropout(head_entity) # 过 dropout size [n,d1]    
        
        # 取出尾实体的表示
        # e_t = self.E(tail_entity) # 维度是[n,d1]，其中n是实体的个数
        # y = self.bn0(tail_entity) # 归一化
        y = self.input_dropout(tail_entity) # 过 dropout
        y = y.view(bsz,head_entity.size(1),1) # 改变size，成了三维的 => [n,d1,1]
        
        # step2. 将核心tensor 和 r 相乘 W * r
        # x size [n,d1]  [d1,d1*d2]
        # W 的size [d2,d1,d1] ，先reshape一下，再计算二者的乘法过程是： [n,d1] * [d1,d2*d1]  => [n,d1*d2]
        W_mat = torch.mm(x, self.W.view(x.size(1), -1))
        W_mat = W_mat.view(bsz, -1,  head_entity.size(1)) # 再reshape 成 [n,d2,d1]
        W_mat = self.hidden_dropout1(W_mat)
        
        # step3. 将 y 同 (W * r) 相乘
        # [n,d2,d1] * [n,d1,n]
        z = torch.bmm(W_mat,y)  # size [n,d2,1] 这个bmm相当于批次的mm，外面的n 是batch size        
        z = z.view(-1,self.W.size(0))
        z = self.bn1(z)
        z = self.hidden_dropout2(z)

        # step4. 再做 (x *(W*r)) * (self.E.weight) 
        # 这里为啥还要做个 x 与 self.E.weight 的乘法？  => 就是用余弦计算相似度，然后使用sigmoid 求某位的最大值
        # [n,n,d2] * [d2,all_entity_num]
        z = torch.mm(z, self.R.weight.transpose(1,0)) 
        # pred = torch.sigmoid(z) # size [n,all_relation_num]
        return z


class BERT_TUCKER(nn.Module):
    ''' Parameters
        config ：
        transr : 已经训练好的kb模型
    '''
    def __init__(self, config):
        super(BERT_TUCKER, self).__init__()
        self.config = config
        self.relation_num = config.relation_num
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
        
        # 是否固定bert？ 即不更新bert的参数
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.gcn_dim = config.gcn_dim
        self.bank_size = self.gcn_dim
        assert self.gcn_dim == config.bert_hid_size + config.entity_id_size + config.entity_type_size        
        
        # 图神经网络之后，紧跟了这么几层用于最后的预测操作        
        kwargs = {} # 下面这几个参数是参考原文获取的
        kwargs['input_dropout'] = 0.3
        kwargs['hidden_dropout1'] = 0.4
        kwargs['hidden_dropout2'] = 0.5
        # d1 表示的是实体的维度，d2 表示的是relation 的维度
        # 这里 d2 的维度取多少合适？ => 按照论文的建议直接取到关系的个数
        self.tucker = TuckER(relation_num=config.relation_num,d1=self.gcn_dim,d2=config.relation_dim,**kwargs)
        self.predict = nn.Linear(in_features=1616,out_features=97)
    def forward(self, **params): # 对一个batch 进行操作
        '''
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]
        distance: [batch_size, max_length]
        entity2mention_table: list of [local_entity_num, local_mention_num]
        mention_graphs: list of DGLHeteroGraph
        entity_graphs: list of DGLHeteroGraph
        h_t_pairs: [batch_size, h_t_limit, 2]
        ht_pair_distance: [batch_size, h_t_limit]
        '''
        input_ids = params['words'] # (batch_size,max_length) 这个input_ids 就是针对输入数据得到的input_ids
        mask = params['mask']
        batch_entity_num = torch.max(params['entity_id'],dim=1)[0] # 获取当前batch中各个doc的 entity num
        max_entity_num = max(batch_entity_num).item() # 求出当前batch的最大 entity num 
        h_t_limit = max_entity_num**2*97 # 三元组的个数，用于后面的padding 操作 
        bsz, slen = input_ids.size()
        # =========================== step 1. 获取原始句子、mention、entity 的表示  ===========================
        # GPU
        # 先过bert.  encoder_outputs 是 last_hidden_states  
        output = self.bert(input_ids=input_ids, attention_mask=mask)
        encoder_outputs = output.last_hidden_state
        sentence_cls = output.last_hidden_state[:,0,:]
        # del output # 删除变量，控制GPU内存
        # torch.cuda.empty_cache()
        # encoder_outputs[mask == 0] = 0
        
        # if self.config.use_entity_type:
        encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type'])], dim=-1)

        # if self.config.use_entity_id:
        encoder_outputs = torch.cat([encoder_outputs, self.entity_id_emb(params['entity_id'])], dim=-1)
        # TODO 在768后面再追加指定的维度，为什么要追加这个表示？？？
        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),
            dim=-1)
        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]
        # sentence_cls: [batch_size, bert_hid+type_size+id_size]

        # =========================== step 2. 获取各个doc中的 mention 表示  ===========================
        mention_id = params['mention_id']  # tokenizer 之后得到的token，在每个位置是否属于mention 的下标
        # 做什么用的？ => 一个batch中的所有doc的所有 mention 拼接在一起得到的表示，这个mention的表示是按照
        # 每个mention所占的 token 加权平均得到的

        # entity2mention_table 这个是一篇doc中 entity的个数 和 mention 的形成的矩阵。矩阵中的值是怎么样形成的？
        # eneity2mention[i][j]=1 表示的就是第i个entity和 第 j 个 mention 是同一个实体，相反，如果没有联系，则为0
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id']) # 选择当前batch 中最大数目的entity 作为 entity_num （在dev_44.json 中，其值为31）
        # 当前定义的变量都是为了后面的程序服务的！！因为后面要用到 entity_num，即需要知道当前批次最大的个数是什么
        # entity_num+1 是因为有一个全局节点[CLS]，如果不要全局节点，那么只用一个entity_num 即可
       
        pred = [] # 用于放置一批数据的预测结果
        # 下面这个for 循环，就是用来获取这个 features
        # features size = [all_mention_num,808] 这里all_mention_num 是一个batch中所有的mention叠加得到的
        for i in range(bsz): # batch是多少，就有几个图。 （针对每篇文档都建有一个图）
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]  拿到每篇doc 中每个token的 embedding 表示
            mention_num = torch.max(mention_id[i]) # 这个参数的含义是：这个doc中 vertexSet中mention 的个数（一篇doc所有entity中的mention）num(mention) >= num(entity)。
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen] select_metrix[i][j] =1代表的就是第i个mention在位置j出现过，否则没有出现。select_metrix.size(1) 就是doc 的长度 
            # average word -> mention  找出每行有几个数。 相当于一个 average pool
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
            # 根据select_metrix 和 encoder_output 做乘法，得到实体的表示
            x = torch.mm(select_metrix, encoder_output)  # [mention_num, bert_hid]
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0) # 加上整个句子的表示。x.size(0) => x.size(0) + 1 。            
            
            # =========================== step 3. 融合形成 entity 表示 ===========================        
            # mention -> entity
            cur_idx = 0        
            # average mention -> entity  # 这个 entity2mention_tabel 本身就是float类型
            # entity2mention_table[i] 表示的就是第i张图（第i篇doc）
            select_metrix = entity2mention_table[i].float() # [local_entity_num+1, mention_num+1]  对每个图（doc）进行操作， 
            # TODO 为什么要赋值？  => 在创建 entity2mention 的时候（entity 和 mention 的index都是从1开始计数），所以0就没有数
            select_metrix[0][0] = 1
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1)) # 做这个扩展操作是为了后面的除法
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)  # 这个操作同上面，也是为了保持每行的值相同 。这里就是为了计算每个下标该有多大的权重去获取最后的entity表示。相当于对 mention 的一个加权            

            # 通过合并mention的特征就能获取到 entity_feature
            # 同理，这个select_metrix.size(0) 就是 entity_num + 1 。
            # 所以这里就相当于给一篇doc中的所有的 entity 赋特征值。
            # select_metrix.size(1) 就是mention_num
            node_num = select_metrix.size(1)

            # 这个entity_representation 是包含了gloabl doc 节点的
            entity_representation = torch.mm(select_metrix, x)
            
            # entity_representation[1:]表示去除 doc 节点，其大小就表示当前的entity的个数； entity_representation[0:]表示包括doc节点
            head_entity = []#entity_representation[1:]
            tail_entity = []#entity_representation[1:]
            cur_pred= []
            for k in range(entity_representation.size(0)):
                if k == 0:
                    continue
                for j in range(entity_representation.size(0)):
                    if j ==0 :
                        continue
                    head_entity.append(entity_representation[k,:])
                    tail_entity.append(entity_representation[j,:])
                    
            head_entity = torch.stack(head_entity,dim=0)
            tail_entity = torch.stack(tail_entity,dim=0)
            # TODO: 将tucker 处理的部分设计成 batch 操作
            cur_pred = self.tucker(head_entity,tail_entity) # size [n,n,r] 表示的是 n个实体两两间的关系
            # cur_pred  = self.predict(torch.cat((head_entity,tail_entity),dim=-1))
            cur_pred = cur_pred.view(-1) # 更改shape，方便后面做 CE_Loss 的计算
            cur_idx += node_num
            # 将cur_pred pad到当前batch的最大长度
            cur_pred = F.pad(cur_pred,(0,h_t_limit-len(cur_pred)))
            pred.append(cur_pred)
        pred = torch.stack(pred,dim=0)
        return pred



class BiLSTM(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config.lstm_hidden_size,
                            num_layers=config.nlayers, batch_first=True,
                            bidirectional=True)
        self.in_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_lengths):
        '''
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        '''

        self.lstm.flatten_parameters()
        bsz, slen, input_size = src.size()

        src = self.in_dropout(src)

        new_src_lengths, sort_index = torch.sort(src_lengths, dim=-1, descending=True)
        new_src = torch.index_select(src, dim=0, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                      padding_value=self.config.word_pad)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=0, index=unsort_index)

        src_h_t = src_h_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        src_c_t = src_c_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
        output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)
        output_h_t = torch.index_select(output_h_t, dim=0, index=unsort_index)
        output_c_t = torch.index_select(output_c_t, dim=0, index=unsort_index)

        outputs = self.out_dropout(outputs)
        output_h_t = self.out_dropout(output_h_t)
        output_c_t = self.out_dropout(output_c_t)

        return outputs, (output_h_t, output_c_t)