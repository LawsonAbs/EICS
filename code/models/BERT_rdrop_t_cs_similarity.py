import math
import inspect
import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import random
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer,BertConfig
from utils import get_cuda,get_labelid2name
import dgl
import dgl.nn.pytorch as dglnn
from transformers import BertModel,BertTokenizer
tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-uncased")
id2name = get_labelid2name()

'''
# 根据传入的实体对关系建图
params:
    h_t_pairs: h_entity, t_entity pair
    entity_
    threshold: 阈值范围
    h_t_type: 记录每对实体中的类型信息
    batch_entity_num: 表示当前这个batch里，各个doc中的entity数
    entity_pair_multi_relation_predict: BERT 模型的预测结果
'''
def create_heterograph(h_t_pairs,entity_pair_multi_relation_predict,threshold,relid2name,h_t_type,batch_entity_num):
    graphs = []
    for cur_batch_idx in range(entity_pair_multi_relation_predict.size(0)):
        pred =  entity_pair_multi_relation_predict[cur_batch_idx]
        pred = torch.sigmoid(pred) # 求出每类别下的概率
        data_dict = defaultdict(list)
        # 第r种关系下，所有的样本对的预测值
        # 关系应该从1开始，而不是0 开始
        for r in range(1,97): 
            idx = torch.nonzero(torch.gt(pred[:,r],threshold)).view(-1)            
            adj_matrix_r = h_t_pairs[cur_batch_idx][idx]
            
            # 过滤掉 adj_matrix_r 中 (h_idx == t_idx) 的实体对
            x = adj_matrix_r[:,0]
            y = adj_matrix_r[:,1]
            mask = (x!=y)
            idx = mask.nonzero().view(-1) # 得到过滤后的下标
            # print(idx.size())
            adj_matrix_r = adj_matrix_r[idx]

            # 如果没有这种关系的实体对，那么就直接使用第一对实体对中的头尾实体 x
            # 如果没有这种关系的实体对，那么就直接使用可能性最大的一堆实体 x 
            # 如果没有这种关系的实体对，那么就下一个 √
            if len(adj_matrix_r.T[0]) == 0:                
                continue
            else:
                data_dict[('h_type', relid2name[r], 'h_type')] = (adj_matrix_r.T[0].view(-1),adj_matrix_r.T[1].view(-1))
    
        
        # 如果上述操作没有得到边信息，则加概率最大的非NA关系的边到其中
        if len(data_dict) == 0:
            # step1.先将 pred 中NA关系> 其他关系的 pair 对过滤掉 => 得到pred_2, h_t_pairs_2
            cur_threshold = pred[:,0].unsqueeze(1)            
            b = pred > cur_threshold            
            b = torch.sum(b,dim=-1)            
            idx_filter = torch.nonzero(b).view(-1)
            # 使用过滤后的下标找出pair 和 a
            pred_2 = pred[idx_filter]
            h_t_pairs_2= h_t_pairs[cur_batch_idx][idx_filter] 
                        
            # step2. 接着使用topk 找出前k个关系
            val_1 ,idx_1 = torch.topk(pred_2,k=1,dim=1) # 这里必须是 k = 1
            # 再从上述的结果中，再找出前k个 pred
            if (len(val_1) > 0): # 如果 val_1 有值，那么就执行选取操作
                # 如果没有 topk 个，那么该怎么处理？ 那么就取 val_1.size(0) 和 5 的较小值
                val2,row_idx = torch.topk(val_1,k=min(5,val_1.size(0)),dim=0) # 实体对下标，可能没有k个怎么办？
                # 从 idx_2 中选择出 idx_1 的值，得到的就是关系下标
                rel_idx = idx_1[row_idx].view(-1)
                new_pair = torch.index_select(h_t_pairs_2,0,row_idx.view(-1))
                if len(rel_idx) > 0:
                    for i in range(len(rel_idx)):
                        h_idx,t_idx = new_pair[i]
                        data_dict[('h_type', relid2name[rel_idx[i].item()], 'h_type')] = (h_idx.view(-1),t_idx.view(-1))            

        
        # 如果最后一条边都没有，那么就象征性的加一条边到其中 x
        # TODO: 加概率最大的k条边到其中
        if len(data_dict) == 0:            
            h_idx = h_t_pairs[cur_batch_idx,0,0]
            t_idx = h_t_pairs[cur_batch_idx,0,1]
            data_dict[('h_type', relid2name[r], 'h_type')] = (h_idx.view(-1),t_idx.view(-1))
        # 同时需要保证创建图的节点数也是每个doc中 entity 的个数，不能漏掉
        num_nodes_dict = {'h_type':batch_entity_num[cur_batch_idx]} # h_type 这种类型的节点的个数 
        graph = dgl.heterograph(data_dict,num_nodes_dict=num_nodes_dict)
        graphs.append(graph)
        
    # 因为batch操作仅支持有相同类型的relation的图，所以这里就不batch，直接返回所有的graph
    # 创建得到的图，如果只有一条边，那么大概率就是 P1198(unemployment rate) 这种关系的边    
    return graphs


''' 
实现的RGCN，用于多关系推理
'''
class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.

    # TODO: 这里的 bases 是啥意思？    
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    # 构造函数中对各个参数进行初始化
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases  # TODO？
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop  
        # dglnn.HeteroGraphConv  是一个在异质图上计算卷积的通用模块
        # 这个定义的方式其实就是RGCN，对不同的 relation 进行聚合操作         
        self.conv = dglnn.HeteroGraphConv({
            # 对不同的 rel 定义一个卷积模块，即 GraphConv
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })
        self.use_weight = weight
        # TODO 下面这个逻辑关系还不是很懂
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters        
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var() ## TODO ？
        if self.use_weight: # rel_name=['intra','inter','global']
            # weight size = [relation_num,feature_size,feature_size]
            weight = self.basis() if self.use_basis else self.weight 
            
            # 得到每种relation 的weight            
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        # 使用 wdict 这个字典中的参数，即作为 $W_r$，执行一个 GraphConv 操作，得到就是
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        # 在计算好的节点表示中，再进一步处理，比如添加自环，添加偏置再做激活和dropout
        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        
        # 返回的是一个字典，这个 ntype 的值就是固定的，为 node
        feat = {ntype: _apply(ntype, h) for ntype, h in hs.items()}        
        return feat



ACT2FN = {
    "relu": nn.functional.relu,
    "gelu": nn.functional.gelu,
    "tanh": torch.tanh,    
    "sigmoid": torch.sigmoid,
}

def get_extended_attention_mask(attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask



# 仅作Bert处理
class BERT_T(nn.Module):
    ''' Parameters
        config ：
    '''
    def __init__(self, config):
        super(BERT_T, self).__init__()
        self.config = config
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."
        
        # 向量初始维度的dim，也就是bert后的维度
        self.emb_dim = 768

        # 下面这两个都是在需要用到的时候，才使用这个去得到具体的数
        if config.use_entity_type:
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                    padding_idx=config.entity_type_pad)
            self.emb_dim += config.entity_type_size
        # 这个 use_entity_id 值得思考，到底是怎么来的？
        if config.use_entity_id:
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)
            self.emb_dim += config.entity_id_size
        
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=config.bert_path,
        attention_probs_dropout_prob=config.bert_dropout,
        hidden_dropout_prob=config.bert_dropout,  
        output_hidden_states=True # 输出每一层的 hidden states
        )
        
        # 是否固定bert？ 即不更新bert的参数
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False
        
        self.bank_size = self.emb_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)
        
        self.bert_encoder = BertEncoder(config=BertConfig(hidden_size=self.emb_dim,
                                                            num_attention_heads=config.t_head_num, 
                                                            num_hidden_layers = config.interaction_layers,
                                                            hidden_dropout_prob = config.interaction_hidden_dropout_prob
                                                            ))
        
        # 过完Bert的预测，得到各个entity pair之间的 multi_relation。
        # self.first_predict = nn.Sequential(            
        #     nn.Linear(self.gcn_dim * 2, self.gcn_dim ),
        #     self.activation,  # TODO: 这个地方的激活函数没有必要用 relu
        #     self.dropout,
        #     nn.Linear(self.gcn_dim , config.relation_num),            
        # )
        # # 图神经网络之后，紧跟了这么几层用于最后的预测操作
        self.predict = nn.Sequential(            
            nn.Linear(self.emb_dim * 6 , self.emb_dim * 2),
            self.activation,
            self.dropout,
            # 引入一个threshold class，所以这里加1
            nn.Linear(self.emb_dim * 2, config.relation_num),
        )
        self.relid2name = get_labelid2name()
        self.cs_weight = config.cs_weight

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
        labels = params['labels']
        # h_t_type = params['h_t_type'] # 得到当前batch的实体类型 => TODO: 这个要在data 中获取
        max_entity_num = torch.max(params['entity_id']) # 获取当前批次所有doc中最大的实体数
        batch_entity_num = torch.max(params['entity_id'],dim=1)[0] # 获取idx
        bsz, slen = input_ids.size()
        h_t_type = torch.zeros(bsz,max_entity_num)
        # =========================== step 1. 获取原始句子、mention、entity 的表示  ===========================        
        # 先过bert.  encoder_outputs 是 last_hidden_states  
        output = self.bert(input_ids=input_ids, attention_mask=mask)
        lst_hidden_states = output.last_hidden_state        
        input_emb = output.hidden_states[0] # 得到原始的输入，使用这个输入的原因是想利用预训练中的常识知识
        encoder_outputs = lst_hidden_states + self.cs_weight*input_emb # 先乘再相加
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
        batch_entity_feateature = torch.zeros(bsz,max_entity_num,self.emb_dim,dtype=torch.float32).cuda()
        # features size = [all_mention_num,808] 这里all_mention_num 是一个batch中所有的mention 数目
        # 上面的 batch_entity_features 的size固定，
        

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
            


        # =========================== step 3. 使用 2层自定义的Transformer 框架 ===========================                
        # 对获取到的entity_features 送到 2层transformer 中进行处理
        # entity_features size = [batch,entity_num,entity_emb_dim]
        # 这里 entity_num 是取当前batch中 最多的entity
        # self.double_transformer(entity_features)        
        entity_mask = torch.zeros(bsz,max_entity_num) # 按照entity_bank的size 得到一个全0 的值
        entity_id = params['entity_id'] # entity_id 是从1 开始计数
        for i in range(bsz): # 对每个batch
            cur_id_num = max(entity_id[i]) # 找出当前batch中每条doc中entity的个数
            entity_mask[i,:cur_id_num] = 1 
        extended_attention_mask = get_extended_attention_mask(attention_mask = entity_mask,input_shape=entity_mask.size()).to(torch.device('cuda:0'))
        # 
        out = self.bert_encoder(batch_entity_feateature,attention_mask = extended_attention_mask)
        fin = out[0] # 得到最后的表示
        
        # 这里的 batch_feature_bert 是list，没有固定的size，追加的顺序是和entity id的顺序相同，
        # 后面直接返回给 entity_graph 的图作输入
        batch_feature_bert = []
        for i in range(bsz): # batch是多少，就有几个图。
            cur_entity_num = batch_entity_num[i].item()
            batch_feature_bert.append(fin[i][0:cur_entity_num])

        # =========================== step 3.  ===========================                
        # 选择当前batch 中最大数目的entity 作为 entity_num （在dev_44.json 中，其值为31）
        # 当前定义的变量都是为了后面的程序服务的！！因为后面要用到 entity_num，即需要知道当前批次最大的个数是什么
        max_entity_num = torch.max(params['entity_id'])

        # =========================== step 4. 获取各个 entity pair的表示，判断entity pair之间的多重关系 ===========================
        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        # h_t_limit = h_t_pairs.size(1)  # 这个参数是什么意思？ => 见上分析

        # 找出 h_entity/t_entity 的index 过程 
        # [batch_size, h_t_limit, bank_size] 为什么要搞成这个形状 => 因为后面要做一个gather操作，需要保证每行的值都是相同的
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.emb_dim)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.emb_dim)
        # [batch_size, h_t_limit, bank_size]

        # 从batch_entity_feat 中得到每个 entity 的表示，称作是h_entity/t_entity
        h_entity = torch.gather(input=batch_entity_feateature, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=batch_entity_feateature, dim=1, index=t_entity_index)
        
        # 得到过完 transformer 的表示
        h_entity_t = torch.gather(input=fin,dim=1,index=h_entity_index)
        t_entity_t = torch.gather(input=fin,dim=1,index=t_entity_index)
        
        # 根据每句话中的 entity pair 来判断关系，得到的是局部的 prediction
        # 根据这预测得到的句内 entity之间的 prediction，然后根据这个 prediction 建图
        # [bsz,entity_pair_num,relation_num]
        # entity_pair_multi_relation_predict = self.first_predict(torch.cat((h_entity_t,t_entity_t),dim=-1))
        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), h_entity_t,t_entity_t),
            dim=-1))
        sim_loss = self.calculate_similarity()

        return  predictions,sim_loss

    # 计算矩阵各个行间的相似度，将其控制在较小的范围值内
    def calculate_similarity(self):
        matrix = self.predict[3].weight
        # step 1. 计算行向量的长度
        len_matrix = torch.sqrt(torch.sum(matrix**2,dim=-1))
        # print(len_matrix)

        b = len_matrix.unsqueeze(1).expand(-1,matrix.size(0))
        c = len_matrix.expand(matrix.size(0),-1)
        
        # step2. 计算乘积
        x = matrix @ matrix.T
        # print(x)

        # step3. 计算最后的结果
        res = x/(b*c) # 相似度矩阵
        # print(res)
        # 这里计算的损失是有问题的，不应该这么计算
        upper_matrix = torch.triu(res,diagonal=1) # 得到上三角矩阵
        
        z = torch.tensor(0,dtype=torch.float32).cuda()

        # 如果大于0，我们就优化它，如果小于0，其实是可以接受的
        sim_loss = torch.sum(torch.where(upper_matrix > 0.5,upper_matrix,z))         
        return sim_loss


class RRCS(nn.Module):
    ''' Parameters
        config ：
    '''
    def __init__(self, config):
        super(RRCS, self).__init__()
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
        
        self.gcn_dim = config.gcn_dim
        
        self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)
        
                
        self.relid2name = get_labelid2name()

        # GAIN 用的 rel_name_lists 是使用 entity-entity, mention-mention 这种关系
        # TODO: 但是在RRCS中，我们使用96种 relation (不包含NA 这种关系)，需要考虑使用 NA 这种关系吗？
        self.rel_name_lists = list(self.relid2name.values()) # 强转成list
        self.RGCN = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, self.rel_name_lists,
                                                           num_bases=len(self.rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=self.config.dropout)
                                         for i in range(config.gcn_layers)])

        # 图神经网络之后，紧跟几层变换做最后的预测操作
        self.predict = nn.Sequential(            
            nn.Linear(self.bank_size * 4 , self.bank_size * 2),
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_num),
        )
        
        

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
        # h_t_type = params['h_t_type'] # 得到当前batch的实体类型 => TODO: 这个要在data 中获取
        max_entity_num = torch.max(params['entity_id']) # 获取当前批次所有doc中最大的实体数
        batch_entity_num = torch.max(params['entity_id'],dim=1)[0] # 获取idx
        bsz, slen = input_ids.size()
        h_t_type = torch.zeros(bsz,max_entity_num)
        entity_graphs = params['entity_graphs']
        batch_feature_bert = params['batch_feature_bert']
        max_entity_num = torch.max(params['entity_id'])

        
        # =========================== step 4. 获取各个 entity pair的表示，判断entity pair之间的多重关系 ===========================
        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)  # 这个参数是什么意思？ => 见上分析


        entity_bank = torch.zeros(bsz,max_entity_num,808*3).cuda()        
        # entity_graphs = entity_graphs.to(batch_feature_bert.device)                
        starttime = datetime.datetime.now()
        # 对每个graph（doc）做RGCN操作，然后得到他们的返回值，将其做为每篇doc中的 entity 的最终表示
        for i in range(bsz):
            cur_feature_bert = batch_feature_bert[i] # 当前 doc 中的实体经过 bert 后的特征
            # 保证特征数（实体节点数）和图节点数相同 => 如果上面的 create_heterograph 的节点数不够，则容易出问题
            # 缓存所有节点的特征
            temp = cur_feature_bert
            for rgcn in self.RGCN:
                # 输入的特征是当前这个图的entity的特征
                out = rgcn(entity_graphs[i],{"h_type":cur_feature_bert})['h_type'] # rgcn on entity_graphs[i]
                temp = torch.cat((temp,out),dim=-1)
            # 从 batch 的图中获取得到各个节点的表示，并放到 entity_bank 中
            cur_entity_num = batch_entity_num[i]            
            entity_bank[i,0:cur_entity_num,:] = temp # 得到图中所有节点的特征
        endtime = datetime.datetime.now()
        # print(f"RGCN 耗时 {(starttime-endtime).microseconds/1000}ms")  
        # =========================== step 6. 获取ht_pairs中谈到的entity feature  ===========================
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size) 
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)
        # 从 entity_bank_after_RGCN 中得到每个 entity 的表示，称作是h_entity/t_entity
        h_entity_final = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity_final = torch.gather(input=entity_bank, dim=1, index=t_entity_index)        
        
        # TODO:  对 h_entity, t_entity 加一个head/tail 的处理

        # =========================== step 5. 获取当前batch中需要的entity_id ===========================
        predictions_rgcn = self.predict(torch.cat(
            (h_entity_final, t_entity_final, torch.abs(h_entity_final - t_entity_final), torch.mul(h_entity_final, t_entity_final)),
            dim=-1))
        
        return  predictions_rgcn


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads # 12

        # 每个attention 的维度大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 64
        # 理论上说，这个all_head_size 不就应该是 hideen_size 吗？
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size) 
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 就是对数据x 进行一个shape的转换操作
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.
    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.
    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.
    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.
    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.
    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked
    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.
    Examples::
        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states
        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO 下面这个参数是用于作什么的？
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention( # 最后去执行BertSelfAttention 中的forward 方法
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output




# 使用两层BertEncoder
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 最后就用2层 BertLayer
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states, # 每次传入 entity feature
        attention_mask=None, 
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        # self.layer 是BertLayer中传入的config参数决定的
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 这个是用于什么？
            if self.gradient_checkpointing and self.training:

                if use_cache:                    
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # 直接执行layer_module 中的forward()
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return ( # 每层的输出
            hidden_states,
            next_decoder_cache,
            all_hidden_states,
            all_self_attentions,
            all_cross_attentions,
        )
