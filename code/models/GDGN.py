import torch.nn.functional as F
# import dgl
# import dgl.nn.pytorch as dglnn
# from dgl.nn import GraphConv
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer,RobertaTokenizer

from utils import get_cuda,get_pair_nodes


class GDGN_BERT(nn.Module):
    ''' Parameters
        config ：
        transr : 已经训练好的kb模型
    '''
    def __init__(self, config,transr):
        super(GDGN_BERT, self).__init__()
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
        
        # 是否固定bert？ 即不更新bert的参数
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == config.bert_hid_size + config.entity_id_size + config.entity_type_size
        # global 这个参数文章中好像没有提及
        rel_name_lists = ['intra', 'inter', 'global']
        # 有几层layer，就执行几次for 循环。 一般 config.gcn_layers = 2
        # 在构造RelGraphConvLayer 的时候，需要使用到的几个组件，所以就放到init()中了
        # num_bases=len(rel_name_lists)，这个
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=self.config.dropout)
                                         for i in range(config.gcn_layers)])

        for i in self.GCN_layers:
            i.cuda() # 移入到GPU中
        # bank_size 为啥是这么计算的？ self.config.gcn_layers 表示的是 gcn 的层数， +1 表示的是初始化的embedding维度。所以这里是 *(self.config.gcn_layers+1)。 这个可以在论文中的3.2下的公式(4)得出
        self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)
        
        # 图神经网络之后，紧跟了这么几层用于最后的预测操作
        self.predict = nn.Sequential(
            # nn.Linear(15392, self.bank_size * 2),
            nn.Linear(self.bank_size * 5 + self.gcn_dim * 4, self.bank_size * 2),
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_num),
        )
        # TODO 查看这个类的定义，是想做什么？ => 跟上面的这个 RelGraphConvLayer 有什么区别？         
        self.edge_layer = RelEdgeLayer(node_feat=self.gcn_dim, 
                                       edge_feat=self.gcn_dim,
                                       activation=self.activation,
                                       dropout=config.dropout
                                       )

        self.path_info_mapping = nn.Linear(self.gcn_dim * 4, self.gcn_dim * 4)

        self.attention = Attention(self.bank_size * 2, self.gcn_dim * 4)
        # 想让 garph_big 中节点的feat特征改变，所以这里我使用 nn.Parameter将其包裹起来作为一个训练参数
        # confit.entity_num = 57083
        # 创建图的时候，需要创建双向图。
        # 这里的图的node_id 就对应 entity_id。所以后面的设置也需要修改
        left_nodes,right_nodes = get_pair_nodes(path = "../data/train.json")
        # 根据(left_nodes, right_nodes) 只能得到
        # 根据训练集标签获取边的信息。如果二者能构成一个三元组，那么就形成一条边
        # num_nodes = 57083 的个数是根据 train.json + dev.json + test.json 三者中的实体获取到的，预先可以统计出来
        # num_nodes = 36473 的个数是根据 train.json 者中的实体获取到的，预先可以统计出来
        self.num_nodes = 57083
        self.graph_big = dgl.graph((left_nodes, right_nodes),num_nodes=self.num_nodes)
        if config.use_global_graph:
            self.ent_embedding = nn.Embedding(self.num_nodes,808).to(torch.device('cuda:0'))
        self.graph_big = dgl.add_reverse_edges(self.graph_big) # 把图改成双向，创建双向边  
        self.graph_big = self.graph_big.to(torch.device('cuda:0'))
        
        # 对图执行entity_id 记号
        entity_id = [i for i in range(self.graph_big.num_nodes())]                
        entity_id = torch.tensor(entity_id,dtype=torch.int32).to(torch.device('cuda:0'))
        # 本想在刚开始就对其进行赋值操作，但是好像没办法对其进行更新。所以转换方式在后面对['feat'] 进行赋值
        # self.graph_big.ndata['feat'] = (self.graph_init_emb(entity_id)).to(torch.device('cuda:0'))
        entity_id = entity_id.view(self.graph_big.num_nodes(),1).to(torch.device('cuda:0'))
        self.graph_big.ndata['entity_id'] = entity_id
        
        # 搞一个卷积层，用于对 entity-level graph 进行一个卷积处理
        self.conv = GraphConv(in_feats=50, 
                            out_feats=10,
                            norm='both',
                            weight=True, # 在聚合特征之前是否过一个线性层
                            bias=True,  # 对输出加一个bias。
                            activation=self.activation, 
                            allow_zero_in_degree=True
                            )

        self.alpha = 0.1 #
        self.beta = 0.9 #
        
        # 使用预训练好的模型的embedding，这里其实暂时只用到了self.ent_embdding
        self.ent_embedding = transr.ent_embedding
        self.rel_embedding = transr.rel_embedding
        
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
        
        if self.config.use_entity_type:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_id_emb(params['entity_id'])], dim=-1)
        # TODO 在768后面再追加指定的维度，为什么要追加这个表示？？？
        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),
            dim=-1)
        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]
        # sentence_cls: [batch_size, bert_hid+type_size+id_size]

        # params['graph'] 是根据训练数据得到的mention-level子图
        mention_graphs = params['mention_graphs'] # list of <dgl.heterograph.DGLHeteroGraph>
        new_mention_graphs = []  # 考虑一下这个图后面哪里在用？ 如果没有在用，那么就删除
        # 将graphs 放到gpu中，因为后面有一串的矩阵操作
        # 将这个操作放到model调用之前去做
        for graph in mention_graphs:
            graph = graph.to(torch.device('cuda:0'))
            new_mention_graphs.append(graph)
        
        # 这里
        # =========================== step 2. 获取各个doc中的 mention 表示  ===========================
        mention_id = params['mention_id']  # tokenizer 之后得到的token，在每个位置是否属于mention 的下标
        # 做什么用的？ => 一个batch中的所有doc的所有 mention 拼接在一起得到的表示，这个mention的表示是按照
        # 每个mention所占的 token 加权平均得到的
        features = None
        # 下面这个for 循环，就是用来获取这个 features
        for i in range(len(new_mention_graphs)): # batch是多少，就有几个图。 （针对每篇文档都建有一个图）
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]  拿到每篇doc 中每个token的 embedding表示
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
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0) # 加上整个句子的表示。x.size(0) => x.size(0) + 1 。第一个维度加一
            # 最后一行代表的是句子的表示
            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        # =========================== step 3. 使用 GCN 更新 mention-level graph ===========================
        # batch()方法的作用：Batch a collection of DGLGraph s into one graph for more efficient graph computation. 
        # graph_batch ：<class 'dgl.heterograph.DGLHeteroGraph'> 将一批graph(graphs)一起处理
        mention_graph_batch = dgl.batch(new_mention_graphs) # 将源码中的 batch_hetero => batch
        
        # 将 new_graphs 从cuda中移除
        # new_graphs = [graph.cpu() for graph in new_graphs] 
        # torch.cuda.empty_cache() # 让 nvidia-smi 中显示内存释放
        output_features = [features]
        # 对图中的节点进行一个卷积操作。图是刚才batch之后得到的graph_batch。TODO 问题是batch之后节点还是顺序存储的吗？
        for GCN_layer in self.GCN_layers:
            # 返回的feature要接收。用于后面做节点分类，这里跟图更新没有关系，图节点表示的更新仍然是靠梯度完成的
            features = GCN_layer(mention_graph_batch, {"node": features})["node"]  # [total_mention_nums, gcn_dim]
            output_features.append(features)
        
        # 将三个feature 拼接到一起。三个feature分别是：Bert的最后一层 + 两层图卷积的结果。 它们的维度都是 [mention_num,808]，最后得到的维度是[mention_num,808*3]
        output_feature = torch.cat(output_features, dim=-1)

        graphs = dgl.unbatch(mention_graph_batch)
        # mention -> entity
        # entity2mention_table 这个是一篇doc中 entity的个数 和 mention 的形成的矩阵。矩阵中的值是怎么样形成的？
        # eneity2mention[i][j]=1 表示的就是第i个entity和 第 j 个 mention 是同一个实体，相反，如果没有联系，则为0
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id']) # 选择当前batch 中最大数目的entity 作为 entity_num （在dev_44.json 中，其值为31）
        # 当前定义的变量都是为了后面的程序服务的！！因为后面要用到 entity_num，即需要知道当前批次最大的个数是什么
        entity_bank = torch.Tensor(bsz, entity_num, self.bank_size).to(torch.device('cuda:0')) 
        # global_info 是干什么的？ => 获取整个文档的信息，这个会放在最后面作为一个特征用于预测
        global_info = torch.Tensor(bsz, self.bank_size).to(torch.device('cuda:0')) 
        

        # =========================== step 4. 融合形成 entity 表示 ===========================
        cur_idx = 0
        entity_graph_feature = None  # 存储一个batch中所有entity_graph 中的entity的特征，对应关系是什么？
        for i in range(len(graphs)):  # 对每个图都进行一次操作
            # average mention -> entity  # 这个 entity2mention_tabel 本身就是float类型
            # entity2mention_table[i] 表示的就是第i张图（第i篇doc）
            select_metrix = entity2mention_table[i].float() # [local_entity_num, mention_num]  对每个图（doc）进行操作， 
            # TODO 为什么要赋值？  => 在创建 entity2mention 的时候（entity 和 mention 的index都是从1开始计数），所以0就没有数
            select_metrix[0][0] = 1  
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1)) # 做这个扩展操作是为了后面的除法
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)  # 这个操作同上面，也是为了保持每行的值相同 。这里就是为了计算每个下标该有多大的权重去获取最后的entity表示。相当于对 mention 的一个加权
            node_num = graphs[i].number_of_nodes('node')  # 这个 node_num 就是 mention_num == select_metrix.size(1)。 和 num_nodes()功能相同

            # 通过合并mention的特征就能获取到 entity_representation
            # 同理，这个select_metrix.size(0) 就是entity_num 。所以这里就相当于给一篇doc中的所有的 entity 赋特征值。 将所有的entity 属性表示放到entity_bank 中
            entity_representation = torch.mm(select_metrix, output_feature[cur_idx:cur_idx + node_num]) 
            # 这个 entity_bank 有什么用？ => 原代码使用这个entity_bank 来获取对应每个位置实体的表示。 entity_graph_feature 是一个串联起来的，相比entity_bank（有规则的shape），entity_graph_feature 的形状受各个batch 数据的影响
            entity_bank[i, :select_metrix.size(0) - 1] = entity_representation[1:]
            global_info[i] = output_feature[cur_idx]
            cur_idx += node_num

            # entity_graph_feature[idx] 中有很多零值，该怎么办？=> mention level graph 中加自环避免
            if entity_graph_feature is None: 
                # self.gcn_dim = 808， 代表的含义就是从倒数第808个开始，即取最后808个数。我们知道，最后808 维度就是图神经网络第二层的输出结果
                # 为啥是从1 开始？ => 因为上面的 entity2mention_table中的entity和mention都是从1开始计数（可能是第0号结点是根节点，所以要排除掉）
                entity_graph_feature = entity_representation[1:, -self.gcn_dim:] 
            else:
                entity_graph_feature = torch.cat((entity_graph_feature, entity_representation[1:, -self.gcn_dim:]),
                                                 dim=0)
        
        # =========================== step 5. 对entity_graph 使用GCN ===========================
        # 为啥还要从 params 中抽取出图？？ => 因为这里得到的是entity_graph， 还需要在它的基础上进行一个路径推理得到最后的结果。 现在对这个entity_graphs 还不是理解，这里的图只是整个batch 的图，是个list
        # 这里的 entity_graph 有 batch_size 个
        # entity_graphs 是个同构图，只有一种边
        entity_graphs = params['entity_graphs']
        entity_graphs_big = dgl.batch(entity_graphs) 
        entity_graphs_big = entity_graphs_big.to(torch.device('cuda:0'))
        # 即使有重复的entity_id，这里仍然当作是不同的节点来处理
        self.edge_layer(entity_graphs_big, entity_graph_feature) # entity_graph_feature.size(0) == entity_graph_big.num_nodes() 
        # 为得到的entity_graphs 赋值，但是该怎么保证这个赋值过程是一一对应的呢？ => 使用id特征，同时记录entity_id
        entity_graphs = dgl.unbatch(entity_graphs_big)


        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)  # 这个参数是什么意思？ => 见上分析

        # TODO 找出 h_entity/t_entity 的index 过程 => 没看懂
        # [batch_size, h_t_limit, bank_size] 为什么要搞成这个形状 => 因为后面要做一个gather操作，需要保证每行的值都是相同的        
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size) 
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)
        # entity_bank = ([batch_size, max(entity_num), 2424])
        # [batch_size, h_t_limit, bank_size]
        # 这个感觉就是聚合同一实体的mention，然后得到每个entity 的表示，称作是h_entity/t_entity
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        path_info = get_cuda(torch.zeros((bsz, h_t_limit, self.gcn_dim * 4)))
        relation_mask = params['relation_mask']
        path_table = params['path_table'] # 理解一下这个 path_table 的生成过程

        # 下面这个操作很慢，是整个过程较为耗时的部分。 应该是为了想使用 path 进行推理        
        for i in range(len(entity_graphs)): # i 表示第i张图
            path_t = path_table[i]  # 第i张图的path 
            for j in range(h_t_limit): # 感觉这个的含义就是 head_entity 和 tail_entity 能组成对的最大限制个数 就叫 h_t_limit
                if relation_mask is not None and relation_mask[i, j].item() == 0:
                    break
                # 根据上面的这个描述，就能理解： h_t_pairs[i,j,0] 就符合逻辑了
                h = h_t_pairs[i, j, 0].item() # [batch_size,930,2]
                t = h_t_pairs[i, j, 1].item()
                # for evaluate
                if relation_mask is None and h == 0 and t == 0:
                    continue
                # 为什么这里要 h+1/t+1 ？ 然后后面又是 val-1？ => 因为在data.py 的create_entity_graph 中创建的path是按照(i+1,j+1) = (val+1) 创建的。所以这里要还原回去，就需要执行 h+1/t+1 。然后真正的节点下标是val-1，而不是val
                # 虽然写成h+1,t+1 但是依然代表的是节点 h->t 
                if (h + 1, t + 1) in path_t:
                    v = [val - 1 for val in path_t[(h + 1, t + 1)]]
                elif (t + 1, h + 1) in path_t:
                    v = [val - 1 for val in path_t[(t + 1, h + 1)]]
                else:
                    print(h, t, v)
                    print(entity_graphs[i].number_of_nodes())
                    print(entity_graphs[i].all_edges())
                    print(path_table)
                    print(h_t_pairs)
                    print(relation_mask)
                    assert 1 == 2

                middle_node_num = len(v)

                if middle_node_num == 0: # 判断是否有中间节点
                    continue                            
                # 这里分成 forward 和 backward 为什么？
                # forward  这里的forward 分成两个方向，一个是从[h] -> [v]，得到的结果是 forward_first；一个是 从 v-> [t]，得到的结果是 forward_second。这就是文中说的两跳关系。 最后得到 h->t 的路径总和
                # 使用for循环遍历middle_node_num 生成的是同个h组成的list
                edge_ids = get_cuda(entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v)) # 获取这条路径上的所有边信息。 edge_ids([],[]) 传入的是两个list，得到边的ids。如果没有边，则报错
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)  # entity_graphs[i].edata['h'] 获取第i张图的边特征（h）
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)]))
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                # backward => 为什么还要再回去一遍？这个就没啥道理了，感觉就像是单纯为了多拼接一点儿向量
                edge_ids = get_cuda(entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v))
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)]))
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                # 竟然就是直接的拼接？
                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                # 下面这个就是文章中3.3 Entity-level Graph Inference Module 部分的(8)-(10) 三个公式对应的部分。使用attention 计算出一条综合的表示
                _, attn_value = self.attention(torch.cat((h_entity[i, j], t_entity[i, j]), dim=-1), tmp_path_info)
                path_info[i, j] = attn_value

            entity_graphs[i].edata.pop('h')
        
        path_info = self.dropout(
            self.activation(
                self.path_info_mapping(path_info)
            )
        )
        # =========================== step 5. 获取当前batch中需要的entity_id ===========================        
        # 为了避免赋值导致的oom问题，这里先获取子图节点，然后给其赋特征值
        cur_idx = 0
        batch_entity_id = params['batch_entity_id'] # 当前这批entiy 在整个train中的(全局)id。 大小是 batch_size [[...],[...]...]
        cur_batch_entity_ids = [ entity_id for entity_ids in batch_entity_id for entity_id in entity_ids ] # 当前这批次数据中的entity_id
        subgraph_node_id = set(cur_batch_entity_ids)        
        h_t_pairs_global = params['h_t_pairs_global']        
        h_t_pairs_global = h_t_pairs_global - 1 #因为之前有+1，所以这里减回去

        # =========================== step 6. 通过采样，获取 subgraph，并对其执行GCN操作 ===========================        
        # 这里需要自己再考虑一下实现细节 => 训练指定节点id组成的子图。但是这个想法其实是有问题的，因为这样会导致很多test中的节点无法训练
        # 如何知道这些节点对应原始节点的关系？这里我用id来表示
        # 得到的子图最后用于添加到 entity 的embedding 中，用于预测
        subgraph = self.graph_big.subgraph(list(subgraph_node_id)).to(torch.device('cuda:0'))

        # 在self.graph_big 中entity_id 和 node_id 是一一对应的，但是在 subgraph 中，entity_id 和 node_id 不是一一对应的。所以在上面我生成了一个对应关系
        subgraph_feat_emb = None # 子图特征向量
        
        # entity_graph_feature.size(0) 和 cur_batch_entity_ids 大小是一一对应的
        # 且下面for循环的 idx 和 cur_batch_entity_ids 中的值是一一对应的，即能保证 entity_id 对应的特征存在 entity_graph_feature[idx]中
        # 这个 entity_graph_feature 中的存储顺序是什么样的？ => 第一个维度是batch，第二个维度是doc，doc中则按照entity的出现顺序进行存储
        assert entity_graph_feature.size(0) == len(cur_batch_entity_ids)
        entity_id2_feature = {}        
        for idx,entity_id in enumerate(cur_batch_entity_ids):
            # TODO 这里面存在一个问题，如果entity_id 出现了多次，那么该怎么取值？ =>这里我直接覆盖了，但是有么有更好的办法？
            entity_id2_feature[entity_id] = entity_graph_feature[idx]
        
        # 获取子图节点特征        
        # 按node_id增序得到对应的entity_id
        for node_id in range(subgraph.num_nodes()):
            entity_id = subgraph.ndata['entity_id'][node_id]
            # 根据 nn.Embedding 得到指定的向量，这个向量是从全局图中获取
            global_graph_node_embedding = self.ent_embedding(entity_id) 
            # 获取得到之后，最好还需要归一化一下？
            global_graph_node_embedding = F.normalize(global_graph_node_embedding,dim=1)
            
            cur_embedding = global_graph_node_embedding
            if subgraph_feat_emb is None:
                subgraph_feat_emb = cur_embedding
            else:
                subgraph_feat_emb = torch.cat((subgraph_feat_emb, cur_embedding),dim=0) # 获取子图节点的特征feat
            
        # 对子图进行特征赋值
        subgraph.ndata['feat'] = subgraph_feat_emb
        # 这里不想添加自环的原因是想：让模型直接对没有边连接的节点输出0值，这样就可以减轻对整个模型的影响
        # （2）subgraph 不应该有自环。（但是经过实验验证，有么有这个自环其实影响不大【在second_entity的维度只有10的情况下】）
        # 【在second_entity的维度为100的情况下】，如果有自环，模型效果则会有点儿差（训练12epoch后在dev_44.json上 也只有Ignore ma_f1 0.2746的效果）；
        # 但如果没有自环，模型效果在epoch 12 训练之后，dev上的效果为Ignore ma_f1 0.3440
        # subgraph = dgl.add_self_loop(subgraph) 
        
        # 卷积计算的结果 out 只对最后的分类有作用。我们不需要将这个计算结果更新成节点特征，节点特征本身应该作为一个待更新的变量，从而去更好的生成这个卷积结果。
        # 生成这个卷积结果之后，利用这个卷积结果去做关系预测
        out = self.conv(subgraph,subgraph.ndata['feat'].to(torch.device('cuda:0')))
        # 在经过GraphConv之后 entiyt_id -> embedding 的映射关系。 是个临时变量，不会占用很大GPU
        entity_id2embedding = {}
        # subgraph 的图因为没有和以前节点的对应关系，所以就导致出现了问题
        for i,embedding in enumerate(out):
            # subgraph.ndata['entity_id'][i] 的含义是：找到 subgraph 中第i个节点的entity_id 特征
            cur_entity_id = subgraph.ndata['entity_id'][i]
            # 如果需要回传梯度的向量，千万不要使用 to_list()，否则就断链
            entity_id2embedding[cur_entity_id.item()] = embedding

        # =========================== step 7. 获取实体对表示，执行预测操作 ===========================
        # 首先根据 entity_id 获取对应的节点表示                        
        batch_h_entity_index = h_t_pairs_global[:, :, 0]
        batch_t_entity_index = h_t_pairs_global[:, :, 1] 
        
        # 存储最后用于计算损失的实体对表示
        second_h_entity = None
        second_t_entity = None
        for h_entity_ids,t_entity_ids in zip(batch_h_entity_index,batch_t_entity_index):
            for h_entity_id,t_entity_id in zip(h_entity_ids,t_entity_ids):
                # 得到对应表示。 为什么这里会有对应不上的错误？ 原因有两个
                # （1）因为 h_entity_index 是局部 entity id， 而 entity_id2embedding 中存储的是全局entity id，所以会有可能对不上。所以需要修改
                # （2）在 data.py中初始化h_t_pairs_global的时候，是用的是0初始化，所以导致后面很多值都是0，但其实本批次中的entity_id 是没有0的
                if h_entity_id.item() ==-1 : # 如果是填充位置，则随机取tensor
                    h_entity_representation = torch.randn((1,10)).to(torch.device('cuda:0'))
                    second_h_entity = torch.cat((second_h_entity, h_entity_representation),dim=0)
                    t_entity_representation = torch.randn((1,10)).to(torch.device('cuda:0'))
                    second_t_entity = torch.cat((second_t_entity, t_entity_representation),dim=0)
                else: # 得到对应表示
                    h_entity_representation = entity_id2embedding[h_entity_id.item()].unsqueeze(0)  
                    if second_h_entity is None:
                        second_h_entity = h_entity_representation
                    else:
                        second_h_entity = torch.cat((second_h_entity, h_entity_representation),dim=0)
                    

                    t_entity_representation = entity_id2embedding[t_entity_id.item()].unsqueeze(0)
                    if second_t_entity is None:
                        second_t_entity = t_entity_representation
                    else:
                        second_t_entity = torch.cat((second_t_entity, t_entity_representation),dim=0)

        # 变shape
        second_h_entity = second_h_entity.view(bsz,h_t_limit,-1)
        second_t_entity = second_t_entity.view(bsz,h_t_limit,-1)
        # 再拼接
        # h_entity = torch.cat((h_entity,second_h_entity),dim=-1)
        # t_entity = torch.cat((t_entity,second_t_entity),dim=-1)
        
        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1) 
        # 将拼接得到tensor 做一个predict操作，这里要对所有 (h_entity,t_entity,relation) 做一个组合判断
        # h_entity 的原始维度是 [batch_size,h_t_limit,2424]
        # torch.cat 之后得到的维度是 [batch_size,h_t_limit,15352-dim(path_info)]
        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), global_info, path_info),
            dim=-1))
        
        # size = [batch_size,h_t_limit,97=realtion_num] 
        return predictions 


class Attention(nn.Module):
    def __init__(self, src_size, trg_size):
        super().__init__()
        self.W = nn.Bilinear(src_size, trg_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, attention_mask=None):
        '''
        src: [src_size]
        trg: [middle_node, trg_size]
        '''

        score = self.W(src.unsqueeze(0).expand(trg.size(0), -1), trg)
        score = self.softmax(score)
        value = torch.mm(score.permute(1, 0), trg)

        return score.squeeze(0), value.squeeze(0)


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


''' TODO 这个类是什么类？
这是作者写的两个类用于来更新图信息的，功能类似于GraphConv
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

    # TODO 这里的 bases 是啥意思？
    # 
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
        # 这个定义的方式有点儿像是对 relation 进行操作。
        self.conv = dglnn.HeteroGraphConv({ 
            # 这里对所有的 rel 都用的是同种卷积模块，即 GraphConv
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
            weight = self.basis() if self.use_basis else self.weight # TODO？ size = [3,808,808]
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict) # 在wdict 这个字典中都执行一个预定义的Conv 操作。但是这个 wdict 是定义在边上的值，不是用节点的值吗？

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
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()} 


'''TODO 这个类同上面这个类的区别是什么？
这个只是对边进行更新，为了生成后面要用到的path_info
'''
class RelEdgeLayer(nn.Module):
    def __init__(self,
                 node_feat,
                 edge_feat,
                 activation,
                 dropout=0.0):
        super(RelEdgeLayer, self).__init__()
        self.node_feat = node_feat # node feature
        self.edge_feat = edge_feat # edge feature
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.mapping = nn.Linear(node_feat * 2, edge_feat)

    def forward(self, g, inputs):
        # g = g.local_var()
        g.ndata['h'] = inputs  # [total_mention_num, node_feat] 简单的赋值操作，为图中的每个节点都赋予特征h
        '''
        apply_edges : update the features of the specified edges by the provied function
            func:使用这个函数去产生新的边特征。参数可以是DGL内置的方法，也可以是自定义的方法
            edges: 待更新表示的边。默认是更新图中的所有边
        '''
        g.apply_edges(lambda edges: {
            'h': self.dropout(self.activation(self.mapping(torch.cat((edges.src['h'], edges.dst['h']), dim=-1))))})
        g.ndata.pop('h')


class Bert():
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_class, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        print(model_path)
        if model_path == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.max_len = 512

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        # assert ids.size(1) < self.max_len
        ids = ids[:, :self.max_len]  # https://github.com/DreamInvoker/GDGN/issues/4
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens)) # 对tokens中的token 执行self.tokenizer.tokenize 操作，生成的结果作为一个list
        subword_lengths = list(map(len, subwords)) # 求出tokenizer 之后得到的总共的subwords的长度
        subwords = [self.CLS] + list(self.flatten(subwords))[:509] + [self.SEP] # 为了防止超过512，所以这里取到509
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 509] = 512
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - subword_ids: A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - token_start_idxs:An array of indices into the list of subwords. See
                doc of subword_tokenize.
            - subwords: A list of subword
            - （暂时忽略）A mask indicating padding tokens.            
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        # 为啥要将subword_ids 转换成numpy格式？
        return subword_ids.numpy(), token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])
