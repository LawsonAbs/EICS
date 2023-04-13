'''
mention-based 的关系抽取
'''
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

        # 下面这两个都是在需要用到的时候，才使用这个去得到具体的数
        if config.use_entity_type:
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)
        # 这个 use_entity_id 值得思考，到底是怎么来的？
        if config.use_entity_id:
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)
        
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=config.bert_path,
        attention_probs_dropout_prob=config.bert_dropout,
        hidden_dropout_prob=config.bert_dropout,  
        output_hidden_states=True # 输出每一层的 hidden states
        )
        self.gcn_dim = config.gcn_dim
        # 是否固定bert？ 即不更新bert的参数
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False
        
        self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)
        
        self.bert_encoder = BertEncoder(config=BertConfig(hidden_size=808,
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
            nn.Linear(self.gcn_dim * 6 , self.gcn_dim * 2),
            self.activation,
            self.dropout,
            # 引入一个threshold class，所以这里加1
            nn.Linear(self.gcn_dim * 2, config.relation_num),
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
        # labels = params['labels']
        # h_t_type = params['h_t_type'] # 得到当前batch的实体类型 => TODO: 这个要在data 中获取
        max_entity_num = torch.max(params['entity_id']) # 获取当前批次所有doc中最大的实体数(从1开始计数)
        max_mention_num = torch.max(params['mention_id']) # 获取当前批次所有doc中最多的mention 数 (从1开始计数)
        batch_entity_num = torch.max(params['entity_id'],dim=1)[0] # 获取 entity 的个数信息
        batch_mention_num = torch.max(params['mention_id'],dim=1)[0] # 获取 mention 个数信息
        bsz, slen = input_ids.size()
        # h_t_type = torch.zeros(bsz,max_entity_num)
        # =========================== step 1. 获取原始句子、mention、entity 的表示  ===========================        
        # 先过bert.  encoder_outputs 是 last_hidden_states  
        output = self.bert(input_ids=input_ids, attention_mask=mask)
        lst_hidden_states = output.last_hidden_state        
        input_emb = output.hidden_states[0] # 得到原始的输入，使用这个输入的原因是想利用预训练中的常识知识
        encoder_outputs = lst_hidden_states + self.cs_weight*input_emb # 先乘再相加
        sentence_cls = output.last_hidden_state[:,0,:]
        pair_limit = max_entity_num * (max_entity_num-1)
        predictions = torch.zeros(bsz,pair_limit,97).to("cuda:0")
        entity2mention_batch = params['entity2mention_batch'] # 得到每个实体idx对应的mention idx
        
        # if self.config.use_entity_type:
        encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type']),self.entity_id_emb(params['entity_id'])], dim=-1)

        # if self.config.use_entity_id:
        # encoder_outputs = torch.cat([encoder_outputs, ], dim=-1)
        # 在768后面再追加指定的维度，为什么要追加这个表示？？？ => 因为想和实体的表示保持一致
        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),dim=-1)

        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]
        # sentence_cls: [batch_size, bert_hid+type_size+id_size]


        # =========================== step 2. 获取各个doc中的 mention 表示 => 进而得到entity的表示  ===========================
        # tokenizer 之后得到的token，在每个位置是否属于mention 的下标
        # 做什么用的？ => 一个batch中的所有doc的所有 mention 拼接在一起得到的表示，这个mention的表示是按照
        # 每个mention所占的 token 加权平均得到的    
        mention_id = params['mention_id']
        # mention -> entity . entity2mention_table 中的entity_idx和 mention_idx都是从1开始计数
        # entity2mention_table 这个是一篇doc中 entity 和 mention 形成的矩阵(其大小是entity_num * mention_num )。矩阵中的值是怎么样形成的？  => 通过数据分析可以获取
        # eneity2mention[i][j]=1 表示的就是 第i个entity和 第j个mention 是同一个实体，相反，如果没有联系，则为0
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
                
        # 获取一个batch中所有篇章中所有 mention 的表示        
        batch_mention_feateature_cls = torch.zeros(bsz,max_mention_num+1,808,dtype=torch.float32).cuda()   # max_mention_num+1 是因为想保留CLS的向量
        # features size = [all_mention_num,808] 这里all_mention_num 是一个batch中所有的mention 数目
        # 上面的 batch_entity_features 的size固定
        
        # batch
        for i in range(bsz): 
            # step1. 得到 mention 的表示
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
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0) # 拼接整个句子的表示 => 其含义是： CLS的表示 + 当前 doc 各个mention的表示 
            batch_mention_feateature_cls[i, :select_metrix.size(0)+1] = x # 得到各个mention的表示，交给后面的transformer 处理

            # # average mention -> entity  # 这个 entity2mention_tabel 本身就是float类型
            # select_metrix = entity2mention_table[i].float()  # [entity_num, mention_num]  对每个图（doc）进行操作， cur_entity2mention_table[i] 表示的就是第i张图（doc）
            # select_metrix[0][0] = 1 
            # mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1)) # 做这个扩展操作是为了后面的除法
            # select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)  # 这个操作同上面，也是为了保持每行的值相同 。这里就是为了计算每个下标该有多大的权重去获取最后的entity表示。相当于对 mention 的一个加权
            # entity_representation = torch.mm(select_metrix, x) # 通过合并mention的特征就能获取到 entity_representation
            # # 同理，这个select_metrix.size(0) 就是entity_num 。所以这里就相当于给一篇doc中的所有的 entity 赋特征值。 将所有的entity 属性表示放到entity_bank 中
            # batch_entity_feateature[i, :select_metrix.size(0) - 1] = entity_representation[1:]
                        


        # =========================== step 3. 使用 n 层自定义的Transformer 促进mention间交互 ===========================
        # 对获取到的 mention_features 送到 2层transformer 中进行处理
        # mention_features size = [batch,max_mention_num,mention_emb_dim]
        # 这里 max_mention_num 是取当前batch中 最多的entity
        mention_mask = torch.zeros(bsz,max_mention_num+1) # 按照 entity_bank的size 得到一个全0 的值
        mention_id = params['mention_id'] # entity_id 是从1 开始计数
        for i in range(bsz): # 对每个batch
            cur_id_num = max(mention_id[i]) + 1 # 找出当前batch中每条doc中entity的个数
            mention_mask[i,:cur_id_num] = 1
        extended_attention_mask = get_extended_attention_mask(attention_mask = mention_mask,input_shape=mention_mask.size()).to(torch.device('cuda:0'))
        
        out = self.bert_encoder(batch_mention_feateature_cls,attention_mask = extended_attention_mask)
        fin = out[0] # 得到最后的表示，这个表示只用来交互，放在最后的残差中使用
        

        # step4.  根据 mention 计算相似度，然后使用 max pool 操作
        for i in range(bsz):
            predictions_doc = [] # 当前doc的 关系
            cur_entity_num = batch_entity_num[i].item() # 当前doc 的实体个数 
            cur_mention_num = batch_mention_num[i] # 当前doc 的mention 个数
            cur_pair_num = cur_entity_num * (cur_entity_num - 1) # 当前doc 会组成的实体对数
            entity2mention = entity2mention_batch[i] # 当前doc 的 entity2mention 情况            
            x = batch_mention_feateature_cls[i][0:cur_mention_num+1] # 使用bert 原12层得到的结果
            x_t = fin[i][0:cur_mention_num+1] # 使用transformer交互后的结果

            # 将所有的 mention 两两组合，得到头尾实体对            
            h_mention = x.repeat(1,x.size(0)).view(-1,x.size(1)) # 
            t_mention = x.repeat(x.size(0),1) # 得到尾实体

            h_mention_t = x_t.repeat(1,x_t.size(0)).view(-1,x_t.size(1)) # 
            t_mention_t = x_t.repeat(x_t.size(0),1) # 得到尾实体
                        
            cur_mention_pair_pred = self.predict(torch.cat(
            (h_mention, t_mention, torch.abs(h_mention - t_mention), torch.mul(h_mention, t_mention),h_mention_t,t_mention_t),
            dim=-1))
            # reshape 成 [n,n,97] ，方便下面取数
            cur_mention_pair_pred = cur_mention_pair_pred.view(cur_mention_num+1,cur_mention_num+1,97)
            
            
            # 根据组合挑选出需要的关系 => entity - level
            for j in range(cur_entity_num):
                # 第 j 个实体对应的 mention idx
                j_mention_idx = entity2mention[j+1]  # 获取第j个实体对应的mention idx ; j+1 是因为 entity2mention 中的idx 是从1开始
                cur_entity_pred = cur_mention_pair_pred[j_mention_idx] # 先取出对应的头实体
                for k in range(cur_entity_num):
                    if j==k:
                        continue
                    # 第 k 个实体对应的 mention idx
                    k_mention_idx = entity2mention[k+1]

                    # 根据j_mention_idx 和 k_mention_idx 获取对应的predictions ， 并执行max pool 操作，以此来作为entity 的predictions
                    cur_entity_pair_pred = cur_entity_pred[:,k_mention_idx,:].view(-1,97) # 再取出对应的尾实体
                    
                    # 接着对 cur_entity_pair_pred 在预测维度做max 操作，直接使用topk 即可
                    # TODO: 不应该对NA关系也取max pool
                    max_pred = torch.topk(cur_entity_pair_pred,dim=0,k=1)[0]
                    max_pred[0,0] = cur_entity_pair_pred[:,0].min()
                    predictions_doc.append(max_pred.view(-1))
            predictions[i,:cur_pair_num] = torch.stack(predictions_doc)

        return  predictions



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
