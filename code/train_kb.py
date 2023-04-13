from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
import json
from transformers import BertModel
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from utils import get_label2id,get_all_entity

''' 
训练图谱常用的方法有很多种，如transe ，conve，transr，rgcn。这里选择使用TransR
（1）但是关系类型有很多种怎么办？ relation是否也要定义embedding表示？该怎么表示？
（2）relation 和 entity 必须是相同的维度吗？
（3）loss = max(0,d_pos-d_neg+margin) 这里该怎么实现？因为max操作不能求导
如果是loss为0，那么对于当前这批样本就不用求导了，否则更新。
'''

# 定义超参
batch_size = 50000
lr = 0.1
negativa_alpha = 4 # 生成负样本的比例
path= "../data/dev_200.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = t.device("cuda:0")
train_epoch = 10
log_step = 100


# 获取训练数据中的三元组关系，用于后续训练KB
def get_all_triplet():
    label2id = get_label2id() # 获取所有的关系数据
    entity2id = get_all_entity(path)     
    examples = [] # 所有的训练样例
    with open(path,'r') as f:
        cont = json.load(f)        
        for doc in tqdm(cont): # 一篇doc            
            cur_positive = defaultdict(list)
            cur_entity_list = doc['vertexSet'] # 获取当前sample的所有实体信息
            if 'labels' not in doc.keys():
                continue
            cur_labels = doc['labels'] # 如果没有labels，则下一条数据
            for i, label in enumerate(cur_labels):
                cur_head = cur_entity_list[label['h']][0]['name']
                cur_tail = cur_entity_list[label['t']][0]['name']
                relation_id = label2id[label['r']]
                head_entity_id = entity2id.get(cur_head)
                tail_entity_id = entity2id.get(cur_tail)
                cur_key = str(head_entity_id)+"_"+str(tail_entity_id)
                cur_positive[cur_key].append(relation_id) # 存在的关系集合
                # head_entity_id 和 tail_entiy_id 具有 relation_id 这种关系
                examples.append((head_entity_id, tail_entity_id,relation_id,1))
            
            # 找出当前doc中的所有实体id
            cur_entity_id = set()
            for entity in cur_entity_list:
                cur_id = entity2id.get(entity[0]['name'])
                cur_entity_id.add(cur_id) 
            cur_entity_id = np.array(list(cur_entity_id)) # 转为numpy，加速
            for head in cur_entity_id:
                for tail in cur_entity_id:
                    if head == tail:
                        continue
                    cur_key = str(head_entity_id)+"_"+str(tail_entity_id)
                    cur_rel = set(cur_positive[cur_key])
                    rel = set([r for r in range(1,97)])
                    rel = rel-cur_rel
                    # 对于负样本来说，它们俩之间没有关系，所以 r 表示第NA类关系
                    for r in rel:
                        # head_entity_id 和 tail_entiy_id 不具有 r 这种关系
                        examples.append(((head, tail,r,0)))

    # random.seed(20) # 设定随机种子，保持每次运行结果相同
    # random.shuffle(examples)

    return examples


'''
（1）定义TransR 类，用于进行知识图谱的预训练
（2）只有类才可以实现 nn.Module ，否则报错
（3）在TransR中结合了计算损失的方法
'''
class TransR(nn.Module):
    # entity_num: 实体个数
    # relation_num: 关系个数
    # emb_dim: 实体/关系 向量的维度
    # out_feat: 投影之后得到的维度
    def __init__(self,entity_num,relation_num,emb_dim,margin,out_feat):
        super(TransR, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num                
        self.ent_embedding = nn.Embedding(num_embeddings=self.entity_num,embedding_dim=emb_dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.relation_num,embedding_dim=emb_dim)
        self.linear = nn.Linear(in_features=emb_dim,out_features=out_feat) # 用于将映射实体的变换
        self.distance = nn.PairwiseDistance(p=2) # L2 距离
        self.margin = t.tensor(margin).to(device)

    def forward(self,head_entity,tail_entity,relation,mask):        
        # 三者的初始embedding 都是随机初始化的
        head_emb = self.ent_embedding(head_entity) # size [batch_size,emb_dim]
        tail_emb = self.ent_embedding(tail_entity)                
        relation_emb = self.rel_embedding(relation)

        # 得到转换后的表示
        head = self.linear(head_emb)
        tail = self.linear(tail_emb) 

        # 将得到的embedding 做一个归一化操作，否则后面计算得到的损失偏的离谱
        # 归一化操作应该放在什么位置？
        head_emb = F.normalize(head_emb, dim=1)
        tail_emb = F.normalize(tail_emb, dim=1)
        relation_emb = F.normalize(relation_emb,dim=1)

        # 得到正样本的下标=> 获取正样本中head，tail 实体的表示
        pos_index = [i for i in range(len(mask)) if mask[i]]
        pos_index = t.tensor(pos_index).to(device) # 转为index        
        pos_head = t.index_select(head,0,pos_index) 
        pos_tail = t.index_select(tail,0,pos_index) 
        pos_rel = t.index_select(relation_emb,0,pos_index)
        
        # 得到负样本的下标 => 获取负样本中 head, tail 实体的表示
        neg_index = [i for i in range(len(mask)) if not mask[i]]
        neg_index = t.tensor(neg_index).to(device) # 转为index                
        neg_head = t.index_select(head,0,neg_index)
        neg_tail = t.index_select(tail,0,neg_index)
        neg_rel = t.index_select(relation_emb,0,neg_index)
        

        dis_pos = self.distance(pos_head+pos_rel,pos_tail) # 计算二者的距离
        dis_neg = self.distance(neg_head+neg_rel,neg_tail)
        
        min_val = t.tensor([0],requires_grad=True,dtype=t.float).to(device)
        if dis_pos.size(0) > 0:
            dis_pos = t.mean(dis_pos) 
        else:
            dis_pos = t.tensor([0]).to(device)
        if dis_neg.size(0) > 0:
            dis_neg = t.mean(dis_neg) # 求出总损失
        else:
            dis_neg = t.tensor([0]).to(device)
        # print(dis_pos,dis_neg)
        return t.max(min_val,dis_pos-dis_neg+self.margin)


class TripleDataset(Dataset):
    def __init__(self,examples):        
        self.examples = examples

    def __getitem__(self,idx):        
        head_entity_ids = t.tensor(self.examples[idx][0])
        tail_entity_ids = t.tensor(self.examples[idx][1])
        relation_ids  = t.tensor(self.examples[idx][2])
        label = t.tensor(self.examples[idx][3])
        
        return head_entity_ids,tail_entity_ids,relation_ids,label

    def __len__(self):
        return len(self.examples)


'''
训练KB
'''
def train(batch_size,lr):
    examples = get_all_triplet() # 获取所有的三元组信息
    
    dataset = TripleDataset(examples)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=8
                            )
    transr = TransR(entity_num,relation_num,emb_dim,margin,50)
    optimizer  = optim.AdamW(transr.parameters(),lr=lr)
    transr = transr.to(device)
    global_step = 0
    step_loss = 0 
    
    for epoch in range(train_epoch):
        for _,x in enumerate(dataloader):
            # mask 用于标识正负样本对
            batch_head_entity_id,batch_tail_entity_id,batch_relation_id,batch_mask = x  # 得到输入数据
            batch_head_entity_id = batch_head_entity_id.cuda()
            batch_tail_entity_id = batch_tail_entity_id.cuda()
            batch_relation_id = batch_relation_id.cuda()
            batch_mask = batch_mask.cuda()
            loss = transr(batch_head_entity_id,batch_tail_entity_id,batch_relation_id,batch_mask)       
            cur_loss = loss.item()
            step_loss += cur_loss
            if global_step % log_step == 0 and global_step:
                print("epoch = ",epoch,", global step = ",global_step,", cur_loss = ",step_loss/log_step)
                step_loss = 0
            global_step+=1
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 梯度回传
            optimizer.step() # 参数更新
    
    print("Finish Training")    
    # （1）直接保存模型  -> 但是如果我只想要其中的某个部分，还是最好保存成各个对应的向量
    model_path = "./checkpoint/transr.pt"
    t.save(transr.state_dict(),model_path)
    
    # （2）将参数保存到具体的文件中 
    # f = open("entity2vec.txt", "w")
    # enb = transr.ent_embeddings.weight.data.cpu().numpy()
    # for i in enb:
    #     for j in i:
    #         f.write("%f\t" % (j))
    #     f.write("\n")
    # f.close()

    # f = open("relation2vec.txt", "w")
    # enb = transr.rel_embeddings.weight.data.cpu().numpy()
    # for i in enb:
    #     for j in i:
    #         f.write("%f\t" % (j))
    #     f.write("\n")
    # f.close()

    # f = open("relation2vec.txt", "w")
    # enb = transr.rel_embeddings.weight.data.cpu().numpy()
    # for i in enb:
    #     for j in i:
    #         f.write("%f\t" % (j))
    #     f.write("\n")
    # f.close()

    
if __name__ == '__main__':    
    train(batch_size,lr)