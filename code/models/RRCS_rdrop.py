import datetime
'''
使用Rdrop
'''
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
from dgl.nn import GraphConv

from transformers import BertTokenizer
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



# 仅作Bert处理
class BERT(nn.Module):
    ''' Parameters
        config ：
    '''
    def __init__(self, config):
        super(BERT, self).__init__()
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
        )
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
        labels = params['labels']
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
        
        
        return  entity_pair_multi_relation_predict,batch_feature_bert



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
