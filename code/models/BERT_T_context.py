import torch.nn.init as init
from torch.nn.parameter import Parameter

'''
使用n层 transformer 对最后得到的entity 进行建模
使用实体对附近的上下文信息
'''
import math
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer,BertConfig
from utils import get_cuda


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
        
        self.bert = BertModel.from_pretrained(config.bert_path)
        
        # 是否固定bert？ 即不更新bert的参数
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False
                
        self.bank_size = 808
        self.dropout = nn.Dropout(self.config.dropout)
        
        # 图神经网络之后，紧跟了这么几层用于最后的预测操作
        self.predict = nn.Sequential(            
            nn.Linear(self.bank_size * 7 , self.bank_size * 2),
            self.activation,
            self.dropout,            
            nn.Linear(self.bank_size * 2, config.relation_num),
        )
        # 定义使用的attention操作  
        self.attention = Attention(self.bank_size * 2, 768)

        self.bert_encoder = BertEncoder(config=BertConfig(hidden_size=808,
                                                            num_attention_heads=config.t_head_num, 
                                                            num_hidden_layers = config.interaction_layers,
                                                            # hidden_dropout_prob = config.interaction_hidden_dropout_prob
                                                            ))
        # （类型，类型，关系） 的表示，用作bias
        self.type_type_relation = nn.Embedding(config.entity_type_num*config.entity_type_num + 1,config.relation_num)

        self.bias = Parameter(torch.empty(50,97,requires_grad=True))
        bound = 1 / math.sqrt(self.bank_size) if self.bank_size > 0 else 0
        init.uniform_(self.bias, -bound, bound)

        self.eps = 1e-30

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
        h_t_pairs: [batch_size, h_t_limit, 2]        
        '''
        input_ids = params['words'] # (batch_size,max_length) 这个input_ids 就是针对输入数据得到的input_ids
        mask = params['mask']
        bsz, slen = input_ids.size()
        # =========================== step 1. 获取原始句子、mention、entity 的表示  ===========================
        # GPU
        # 先过bert.  encoder_outputs 是 last_hidden_states  
        output = self.bert(input_ids=input_ids, attention_mask=mask,output_attentions= True)
        encoder_outputs = output.last_hidden_state
        sentence_cls = output.last_hidden_state[:,0,:]
        
        if self.config.use_entity_type:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_id_emb(params['entity_id'])], dim=-1)
        # TODO: 在768后面再追加指定的维度，为什么要追加这个表示？？？ => 加上实体类型和id
        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),
            dim=-1)
        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]
        # sentence_cls: [batch_size, bert_hid+type_size+id_size]
                
        # =========================== step 2. 获取各个doc中的 mention 表示  ===========================
        mention_id = params['mention_id']  # tokenizer 之后得到的token，在每个位置是否属于mention 的下标
        # 做什么用的？ => 一个batch中的所有doc的所有 mention 拼接在一起得到的表示，这个mention的表示是按照
        # 每个mention所占的 token 加权平均得到的
        features = None
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
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0) # 加上整个句子的表示。x.size(0) => x.size(0) + 1 。第一个维度加一
            # 最后一行代表的是句子的表示
            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        # =========================== step 4. 融合形成 entity 表示 ===========================        
        # mention -> entity
        # entity2mention_table 这个是一篇doc中 entity的个数 和 mention 的形成的矩阵。矩阵中的值是怎么样形成的？
        # eneity2mention[i][j]=1 表示的就是第i个entity和 第 j 个 mention 是同一个实体，相反，如果没有联系，则为0
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id']) # 选择当前batch 中最大数目的entity 作为 entity_num （在dev_44.json 中，其值为31）
        # 当前定义的变量都是为了后面的程序服务的！！因为后面要用到 entity_num，即需要知道当前批次最大的个数是什么
        # entity_num+1 是因为有一个全局节点[CLS]，如果不要全局节点，那么只用一个entity_num 即可
        # entity_bank = torch.Tensor(bsz, entity_num, self.bank_size).to(torch.device('cuda:0'))  # 
        entity_bank = torch.zeros(bsz, entity_num, self.bank_size).to(torch.device('cuda:0')) 
        # global_info 是干什么的？ => 获取整个文档的信息，这个会放在最后面作为一个特征用于预测
        global_info = torch.Tensor(bsz, self.bank_size).to(torch.device('cuda:0')) 
        
        cur_idx = 0        
        for i in range(bsz):  # 对每个图都进行一次操作
            # average mention -> entity  # 这个 entity2mention_tabel 本身就是float类型
            # entity2mention_table[i] 表示的就是第i张图（第i篇doc）
            select_metrix = entity2mention_table[i].float() # [local_entity_num+1, mention_num+1]  对每个图（doc）进行操作， 
            # TODO 为什么要赋值？  => 在创建 entity2mention 的时候（entity 和 mention 的index都是从1开始计数），所以0就没有数
            select_metrix[0][0] = 1
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1)) # 做这个扩展操作是为了后面的除法
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)  # 这个操作同上面，也是为了保持每行的值相同 。这里就是为了计算每个下标该有多大的权重去获取最后的entity表示。相当于对 mention 的一个加权            

            # 通过合并mention的特征就能获取到 entity_feature
            # 同理，这个select_metrix.size(0) 就是 entity_num + 1 。所以这里就相当于给一篇doc中的所有的 entity 赋特征值。 将所有的entity 属性表示放到entity_bank 中
            # select_metrix.size(1) 就是mention_num
            node_num = select_metrix.size(1)

            # 这个entity_representation 是包含了gloabl doc 节点的
            entity_representation = torch.mm(select_metrix, features[cur_idx:cur_idx + node_num])
            # 这个 entity_bank 有什么用？ => 原代码使用这个entity_bank 来获取对应每个位置实体的表示。 entity_features 是一个串联起来的，相比entity_bank（有规则的shape），entity_features 的形状受各个batch 数据的影响
            
            # entity_representation[1:]表示去除 doc 节点，其大小就表示当前的entity的个数； entity_representation[0:]表示包括doc节点
            entity_bank[i, :select_metrix.size(0)-1 ] = entity_representation[1:] 
            global_info[i] = features[cur_idx]
            cur_idx += node_num           
        
        # =========================== step 3. 使用 2层自定义的Transformer 框架 ===========================                
        # 对获取到的entity_features 送到 2层transformer 中进行处理
        # entity_features size = [batch,entity_num,entity_emb_dim]
        # 这里 entity_num 是取当前batch中 最多的entity
        # self.double_transformer(entity_features)
        entity_mask = torch.zeros(bsz,entity_num) # 按照entity_bank的size 得到一个全0 的值
        entity_id = params['entity_id'] # entity_id 是从1 开始计数
        for i in range(bsz): # 对每个batch
            cur_id_num = max(entity_id[i]) # 找出当前batch中每条doc中entity的个数
            entity_mask[i,:cur_id_num] = 1 
        extended_attention_mask = get_extended_attention_mask(attention_mask = entity_mask,input_shape=entity_mask.size()).to(torch.device('cuda:0'))
        # 
        out = self.bert_encoder(entity_bank,attention_mask = extended_attention_mask)
        fin = out[0] # 得到最后的表示
        # =========================== step 5. 获取ht_pairs中谈到的entity feature ===========================
        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)  # 这个参数是什么意思？ => 见上分析
        
        # [batch_size, h_t_limit, bank_size] 为什么要搞成这个形状 => 因为后面要做一个gather操作，需要保证每行的值都是相同的        
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size) 
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)
        # entity_bank = ([batch_size, max(entity_num), 2424])
        # [batch_size, h_t_limit, bank_size]
        # 从entity_bank 中得到每个 entity 的表示，称作是h_entity/t_entity
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        h_entity_t = torch.gather(input=fin,dim=1,index=h_entity_index)
        t_entity_t = torch.gather(input=fin,dim=1,index=t_entity_index)
        # =========================== step 5. 获取当前batch中需要的entity_id ===========================
        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1) 
        
        # last_layer_attention_scores = output.attentions[-1] # 拿出最后一层的注意力分数
        # # 对各个 head 的分数求和
        # last_layer_attention_scores = torch.sum(last_layer_attention_scores,dim=1)
        # attention_length = last_layer_attention_scores.size(-1)
        entity_pair_context_mask = params['entity_pair_context_mask'][:,:,0:encoder_outputs.size(1)] # 获取各个实体对的上下文位置 [batch_size,ht_pair_num,attention_length]
        # 得到上下文相关信息，然后再归一化一下
        # 防止除0 报nan 错 
        context_info = torch.bmm(entity_pair_context_mask,encoder_outputs) / (torch.sum(entity_pair_context_mask,dim=-1) + self.eps).unsqueeze(-1)


        # 将拼接得到tensor 做一个predict操作，这里要对所有 (h_entity,t_entity,relation) 做一个组合判断
        # h_entity 的原始维度是 [batch_size,h_t_limit,2424]
        # torch.cat 之后得到的维度是 [batch_size,h_t_limit,15352-dim(path_info)]        
        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), context_info,h_entity_t,t_entity_t),
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



class Bert():
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_class, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        print(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
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
