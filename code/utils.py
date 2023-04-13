from curses import window
import os
import copy
from collections import Counter, defaultdict
from datetime import datetime
import json
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import torch
import random

from transformers import BertTokenizer

# 较少数据的几个label
less_label = ['P54','P40','P30','P3373','P50','P69','P400','P26','P1441','P1001','P607','P22','P159','P57','P178','P170','P1344','P6','P127','P20','P108','P206','P156','P710','P155','P118','P166','P276','P123','P194','P674','P58','P1412','P449','P800','P179','P140','P35','P706','P37','P162','P136','P580','P241','P937','P31','P112','P585','P403','P137','P749','P355','P36','P205','P176','P272','P172','P576','P86','P279','P1376','P171','P25','P364','P488','P740','P582','P840','P676','P1056','P1366','P551','P1336','P39','P1365','P737','P190','P1198','P807']

more_label = ['P17','P131','P27','P150','P577','P175','P569','P570','P527','P161','P361','P264','P495','P19','P571','P463','P102']

# 实体类型到id的映射
entity_type_name2id = {"OTH":0,'LOC': 1, 'TIME': 2, 'MISC': 3, 'PER': 4, 'ORG': 5, 'NUM': 6} 


def get_labelid2name():
    path = "../data/rel2id.json"
    with open(path,'r') as f:
        cont = json.load(f)
    id2name = {}
    for key,val in cont.items():
        id2name[val] = key
    
    """
    id2name={79: 'P1376', 27: 'P607', 73: 'P136', 63: 'P137', 2: 'P131', 11: 'P527', 38: 'P1412', 33: 'P206', 77: 'P205', 52: 'P449', 34: 'P127', 49: 'P123', 66: 'P86', 85: 'P840', 72: 'P355', 93: 'P737', 84: 'P740', 94: 'P190', 71: 'P576', 68: 'P749', 65: 'P112', 40: 'P118', 1: 'P17', 14: 'P19', 19: 'P3373', 42: 'P6', 44: 'P276', 24: 'P1001', 62: 'P580', 83: 'P582', 64: 'P585', 18: 'P463', 87: 'P676', 46: 'P674', 10: 'P264', 43: 'P108', 17: 'P102', 81: 'P25', 3: 'P27', 26: 'P26', 37: 'P20', 30: 'P22', 0: 'Na', 95: 'P807', 51: 'P800', 78: 'P279', 88: 'P1336', 5: 'P577', 8: 'P570', 15: 'P571', 36: 'P178', 55: 'P179', 75: 'P272', 35: 'P170', 80: 'P171', 76: 'P172', 6: 'P175', 67: 'P176', 91: 'P39', 21: 'P30', 60: 'P31', 70: 'P36', 58: 'P37', 54: 'P35', 31: 'P400', 61: 'P403', 12: 'P361', 74: 'P364', 7: 'P569', 41: 'P710', 32: 'P1344', 82: 'P488', 59: 'P241', 57: 'P162', 9: 'P161', 47: 'P166', 20: 'P40', 23: 'P1441', 45: 'P156', 39: 'P155', 4: 'P150', 90: 'P551', 56: 'P706', 29: 'P159', 13: 'P495', 53: 'P58', 48: 'P194', 16: 'P54', 28: 'P57', 22: 'P50', 86: 'P1366', 92: 'P1365', 69: 'P937', 50: 'P140', 25: 'P69', 96: 'P1198', 89: 'P1056'}
    """
    return id2name



def get_labelid2name_2():
    path = "../data/rel2id.json"
    with open(path,'r') as f:
        cont = json.load(f) # rel -> id
    rel_info_path = "../data/rel_info.json"
    with open(rel_info_path,'r') as f:        
        line = f.readline()
    label_map = json.loads(line) # rel -> name
    
    id2name = {}
    for key,val in cont.items():
        id2name[val] = key
    
    for key,val in cont.items():
        if key in label_map.keys():
            id2name[val] = label_map[key]

    
    out_path = "../data/rel_id2rel_name.json"
    id2name = dict(sorted(id2name.items(),key= lambda x:x[0]))
    with open(out_path,'w') as f:
        json.dump(id2name,f)
    # {1:"country",...}
    return id2name 



def get_label2id():
    path = "../data/rel2id.json"
    with open(path,'r') as f:
        cont = json.load(f)
    return cont


def get_label_map(rel_info_path):
    with open(rel_info_path,'r') as f:
        line = f.readline()
    label_map = json.loads(line) # 让str类型的数据变成dict
    # print(label_map) # label_map 是一个 dict类型
    return label_map


# 返回一个tensor的cuda版本
def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    # return tensor # 宁愿报错，也拒绝返回CPU


def logging(s):
    print(datetime.now(), s)

# 用于计算精确度的值的指标，主要是计算正负样本，以及平均样本的精确度
class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


'''
获取指定路径文件中所有的entity，然后得到一个 entity2id的map.
之所以给出参数，是因为不同的地方可能使用的范围不同，所以加了参数
'''
def get_all_entity(path):
    entity2id = {} # name => id
    idx = 0
    output_path = path+"_entity2id.json" # 输出内容到指定文件
    if os.path.exists(output_path): # 如果存在当前这个文件，则直接读取
        with open(output_path,'r') as f:
            entity2id = json.load(f)
        return entity2id
    with open (path,'r') as f:
        cont = json.load(f)
    for doc in cont:
        title, entity_list = doc['title'], doc['vertexSet']
        for mentions in entity_list:                
            for mention in mentions:
                name = mention['name']                    
                if name not in entity2id.keys():
                    entity2id[name] = idx
                    idx+=1
    # train.json中有 36473 个不同的entity    
    # all.json 中有 57083 个entity
    # print(len(entity2id))    
    # 将结果文件写成json
    with open(output_path,'w') as f:
        json.dump(entity2id,f)
            
    return entity2id # 返回最后得到的entity2id

# 用于计算一个模型的参数量
def print_params(model):
    # step1.过滤掉可导参数
    # step2.将其变成list格式，然后计算每个矩阵的个数，然后求和。得到最后的值
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))

# 将三个文件合并。train.json + dev.json + test.json
def combine(input_path_1,input_path_2,output_path):    
    # train_path = "/home/lawson/program/RRCS/data/train.json"
    # dev_path = "/home/lawson/program/RRCS/data/dev.json"
    # test_path = "/home/lawson/program/RRCS/data/test.json"
    with open(input_path_1,'r') as f:
        input_1_cont = json.load(f)

    with open(input_path_2,'r') as f:
        input_2_cont = json.load(f)
        
    # with open(test_path,'r') as f:
    #     test_cont = json.load(f)
    all_cont = input_1_cont + input_2_cont
    random.shuffle(all_cont)
    with open(output_path,'w') as f:
        json.dump(all_cont, f)


"""
转换整个训练数据的格式，成为一个较为直观的样子，转换后的格式如下：
doc[原样本中所有得sent拼凑成一个doc]
head entiy , end entity, relation
# 获取所有成对的节点信息 （其实就是标签信息）
"""
def get_pair_nodes(path):
    # 只添加训练集的标签数据，dev的不添加
    # 我这么做的原因是：只用用的path="../data/all.json"时代码在dev上的效果有0.5，但是在test上只有0.1，所以我怀疑是
    # dev 的数据信息泄漏导致    
    rel_info_path = '../data/rel_info.json'
    label_map = get_label_map(rel_info_path) # 获取所有的关系数据
    entity2id = get_all_entity(path)
    labels = set() # 存放所有的label
    left_node = [] # 具有边关系的左节点集合
    right_node = [] # 具有边关系的右节点集合
    with open(path,'r') as f:
        cont = json.load(f)        
        for dic in cont:            
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息            
            if 'labels' not in dic.keys(): 
                continue
            cur_labels = dic['labels'] # 如果没有labels，则下一条数据
            for i, label in enumerate(cur_labels):
                head_entity = cur_entity[label['h']][0]['name']
                end_entity = cur_entity[label['t']][0]['name']
                relation = label_map[label['r']]                                
                line = head_entity + ', ' + end_entity +', ' + relation
                head_entity_id = entity2id.get(head_entity)
                end_entity_id = entity2id.get(end_entity)
                if line not in labels:
                    labels.add(line)
                    left_node.append(head_entity_id)
                    right_node.append(end_entity_id)            
    # 返回具有边的节点信息
    return (left_node,right_node)


'''
分析dev中的数据有多少出现在train中
'''
def analysis_dev_in_train():
    dev_path = "../data/dev.json"
    golden_path = "../data/train.json"
    rel_info_path= '../data/rel_info.json'
    label_map = get_label_map(rel_info_path) # 获取所有的关系数据
    # step1.找出train.json中的所有标签构成golden
    train_labels = set() # 
    with open(golden_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic
            # 拼凑得到一个triplet
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息
            labels = dic['labels']
            for label in labels:
                head_entity = cur_entity[label['h']][0]['name']
                tail_entity = cur_entity[label['t']][0]['name']
                relation = label_map[label['r']]
                cur_triplet = head_entity +"," +tail_entity +","+relation
                train_labels.add(cur_triplet) # 将当前的标签放入其中

    # step2.获取dev.json 的标签
    dev_labels = set()
    with open(dev_path,'r') as f:
        cont = json.load(f)
        for dic in cont:
            # 拼凑得到一个triplet
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息
            labels = dic['labels']
            for label in labels:
                head_entity = cur_entity[label['h']][0]['name']
                tail_entity = cur_entity[label['t']][0]['name']
                relation = label_map[label['r']]
                cur_triplet = head_entity +"," +tail_entity +","+relation
                dev_labels.add(cur_triplet) # 将当前的标签放入其中
    
    # step3.查看二者重合度
    cnt = 0
    same = []
    for label in dev_labels:
        # 拼凑得到一个triplet        
        if label in train_labels:
            cnt+=1
            same.append(label)
    print(cnt) # 有1000条数据是在既出现在dev中，也出现在train中的，占dev的10%

    output_path = "../data/same.txt"
    with open(output_path,'w') as f:
        for line in same: # line 是个dict            
            f.write(line+"\n")

"""
仅通过字典进行预测
"""
def predict_by_dict():
    test_path = "../data/test.json"
    train_path = "../data/train.json"
    rel_info_path= '../data/rel_info.json'
    label_map = get_label_map(rel_info_path) # 获取所有的关系数据
    # step1.找出train.json中的所有标签构成golden
    train_labels = set() # 
    with open(train_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic
            # 拼凑得到一个triplet
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息
            labels = dic['labels']
            for label in labels:
                head_entity = cur_entity[label['h']][0]['name']
                tail_entity = cur_entity[label['t']][0]['name']
                relation = label_map[label['r']]
                cur_triplet = head_entity +"," +tail_entity +","+relation
                train_labels.add(cur_triplet) # 将当前的标签放入其中

    # step2.预测 test.json 
    test_labels = []
    with open(test_path,'r') as f:
        cont = json.load(f)
        for dic in cont:
            # 拼凑得到一个triplet
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息            
            title = dic['title']
            for h_idx,h_entity in enumerate(cur_entity):
                for t_idx,t_entity in enumerate(cur_entity):
                    if h_idx == t_idx:
                        continue
                    head_entity_name = h_entity[0]['name']
                    tail_entity_name = t_entity[0]['name']                    
                    for key,val in label_map.items():
                        temp = head_entity_name + "," + tail_entity_name +","+val
                        if temp in train_labels:
                            cur_labels = {"title":title, "h_idx":h_idx,"t_idx":t_idx,"r":key}
                            test_labels.append(cur_labels) # 将当前的标签放入其中
        
    output_path = "../data/test_pred_by_dict.json"
    with open(output_path,'w') as f:
        json.dump(test_labels, f)


"""
转换整个训练数据的格式，成为一个较为直观的样子，转换后的格式如下：
doc[原样本中所有得sent拼凑成一个doc]
head entiy , end entity, relation
"""
def convert_train_data(path,rel_info_path,out_path):
    label_map = get_label_map(rel_info_path) # 获取所有的关系数据
    all_doc = [] # 所有的文档
    with open(path,'r') as f:
        cont = json.load(f)        
        for dic in cont:
            sample = [] # 一条样例的数据
            cur_sentencs = dic['sents'] # 获取当前sample的所有sentence
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息
            cur_labels = dic['labels']            
            sample.append("##title")
            sample.append(dic['title'])
            sample.append("##doc")
            # step1. 处理文档中的句子
            doc = [] # 一篇文档
            for idx,sentence in enumerate(cur_sentencs):
                cur_str = " ".join(sentence) 
                sample.append(str(idx)+" "+cur_str)
            # temp = " ".join(doc) # 得到该sample的doc
            # sample.append(doc)

            sample.append("##labels")
            # step2. 处理文档中的title
            labels = []
            for i, label in enumerate(cur_labels):
                head_entity = cur_entity[label['h']][0]['name']
                end_entity = cur_entity[label['t']][0]['name']
                relation = label_map[label['r']]
                evidence = label['evidence']
                
                line = head_entity + ', ' + end_entity +', ' + relation +'('+label['r']+"), "+ str(evidence)
                labels.append(line) # 放入当前这个doc中
            sample.extend(labels)
            all_doc.append(sample)
    
    
    with open(out_path,'w') as f:
        for doc in all_doc:
            for line in doc:
                f.write(str(line)+"\n")
            f.write("------------------------\n") # 每个doc之后换行


"""
获取dev的标签在train中出现过的数目，尤其需要注意第7类数据，查看有多少是重叠的？
"""
def get_dev_in_train_num():
    train_path = "../data/train.json"
    dev_path = "../data/dev_44_visual.txt"
    label_map = get_label_map(rel_info_path) # 获取所有的关系数据
    train = set()
    with open(train_path,'r') as f:
        cont = json.load(f)        
        for dic in cont:
            sample = [] # 一条样例的数据
            cur_sentencs = dic['sents'] # 获取当前sample的所有sentence
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息
            cur_labels = dic['labels']            
            
            # step2. 处理文档中的title
            
            for i, label in enumerate(cur_labels):
                head_entity = cur_entity[label['h']][0]['name']
                end_entity = cur_entity[label['t']][0]['name']
                relation = label_map[label['r']]
                evidence = label['evidence']
                
                line = head_entity + ', ' + end_entity +', ' + relation +', '+ str(evidence)
                train.add(line)             
    # train.append("Susan Blue Parsons, United States, country of citizenship,")
    dev = []
    cnt = 0
    with open(dev_path,'r') as f:
        lines = f.readlines() # 每行是一个item
        for line in lines:        
            if "（7）" in line: 
                line = line.split()
                temp = " ".join(line[1:])
                temp = temp.split(",")
                res = ",".join(temp[0:3])
                for i in train:
                    if res in i:
                        cnt+=1
    print(cnt)



"""
分析重复的实体
1.判断test的实体有多少在train中出现过
2.判断dev的实体有多少在train 中出现过
"""
def analysis_repetitive_entity():
    dev_path = "../data/dev.json"
    test_path = "../data/test.json"
    train_path = "../data/train.json"
    
    # step1. 获取 train.json 中的实体
    train_entity = set()
    with open(train_path,'r') as f:
        cont = json.load(f)
        for doc in cont:
            title, entity_list = doc['title'], doc['vertexSet']
            for mentions in entity_list:
                for mention in mentions:
                    name = mention['name'] # 仅根据第一个mention的name 来获取                    
                    train_entity.add(name)
    
    test_in_train_cnt = 0
    test_tot = 0
    with open(test_path,'r') as f:
        cont = json.load(f)
        for doc in cont:
            title, entity_list = doc['title'], doc['vertexSet']
            for mentions in entity_list:
                for mention in mentions:
                    name = mention['name'] # 仅根据第一个mention的name 来获取                
                    if name in train_entity:
                        test_in_train_cnt += 1
                    test_tot+=1
    
    dev_in_train_cnt = 0
    dev_tot = 0
    with open(dev_path,'r') as f:
        cont = json.load(f)
        for doc in cont:
            title, entity_list = doc['title'], doc['vertexSet']
            for mentions in entity_list:
                for mention in mentions:
                    name = mention['name'] # 仅根据第一个mention的name 来获取                
                    if name in train_entity:
                        dev_in_train_cnt += 1
                    dev_tot+=1
    print("test entity in train =",test_in_train_cnt,"tot=",test_tot)
    print("dev entity in train =",dev_in_train_cnt,"tot=",dev_tot)
    
    

# 分析数据中的EPO问题是否足够多？
def analysis_EPO_num( path ):
    label_map = get_label_map(rel_info_path) # 获取所有的关系数据    
    tot_epo = 0
    tot_label = 0
    overlap_relation=[] # 重复关系的累积
    with open(path,'r') as f:
        cont = json.load(f)
        for dic in cont:
            cur_entity = dic['vertexSet'] # 获取当前sample的所有实体信息
            cur_labels = dic['labels']
            tot_label += len(cur_labels)
            labels = {}
            epo_cnt = 0 # 对epo的个数计数（只累积单篇文档中）
            for i, label in enumerate(cur_labels):
                head_entity = cur_entity[label['h']][0]['name']
                end_entity = cur_entity[label['t']][0]['name']
                relation = label_map[label['r']]
                
                key = head_entity + ',' + end_entity +','
                if key in labels.keys() and relation!=labels[key]:
                    epo_cnt+=1
                    overlap_relation.append(relation)
                else:
                    labels[key] = relation
            
            tot_epo += epo_cnt
    
    overlap_relation_map = Counter(overlap_relation)
    print("tot_labels = ",tot_label,",tot_epo = ",tot_epo)
    # a = sorted(overlap_relation_map,key= lambda x:x[1])
    # print(a)
    for item in overlap_relation_map.items():
        print(item)


'''

1.功能：本方法在于获取遗漏的dev 标签 
2.参数：
01.输入
dev_pred表示dev的预测值，dev_golden表示dev的标准label。
02.输出

'''
# 找出pred中遗漏项
def get_pred_omit(pred_path,golden_path,dev_44_visual):
    relation_label2id = get_label2id() 
    relation_id2label = dict(zip(relation_label2id.values(), relation_label2id.keys()))
    rel_info_path = '../data/rel_info.json'    
    relation_id2name = get_label_map(rel_info_path) # relation id -> name
    # ================== step 1.根据golden获取数据 ==================
    title_idx2entity_name = {}
    title2doc={} # 记录title到doc的信息
    with open(golden_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont:
            cur_title = dic['title']
            entity_list = dic['vertexSet']
            for idx,entity in enumerate(entity_list):                
                title_idx2entity_name[cur_title+str(idx)] = entity[0]['name']
            sent = dic["sents"] # 获取title对应的doc
            title2doc[cur_title] = sent

    # ================== step 2.获取golden ==================
    golden = defaultdict(list) # 键不存在的时候直接用list代替
    with open(golden_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic
            cur_labels = dic['labels']
            cur_title = dic['title']
            for cur_label in cur_labels:
                # 拼凑得到一个triplet
                triplet = (cur_label['h'],cur_label['t'],relation_label2id[cur_label['r']])
                golden[cur_title].append(triplet) # 加入一个三元组

    # ================== step 3.找出pred ==================
    # "{"title":[],"title":[]...}"
    pred = defaultdict(list) 
    with open(pred_path,'r') as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic            
            # 拼凑得到一个triplet
            title = dic['title']
            if type(dic['r_idx']) == str and 'P' in dic['r_idx']:
                triplet = (dic['h_idx'],dic['t_idx'],relation_label2id[dic['r_idx']])
            else:
                triplet = (dic['h_idx'],dic['t_idx'],dic['r_idx'])            
            pred[dic['title']].append(triplet) # 加入一个三元组

    omit = dict()
    # ================== step 4.找出遗漏值    ==================
    for title,labels in golden.items():
        # 拼凑得到一个triplet
        if title not in pred.keys(): # 如果doc没有预测值
            omit[title] = labels

        else: # 如果doc有预测值
            cur_pred = pred[title]
            cur_gold = golden[title]
            omit[title]= sorted(set(cur_gold)-set(cur_pred))

    # ================== step 5.找出人为标记    ==================
    out_visual = []
    
    with open(dev_44_visual,'r') as f:
        lines = f.readlines() # 每行是一个item
        for line in lines:
            if "（" in line and "）" in line:
                out_visual.append(line)

    omit = dict(sorted(omit.items(),key = lambda x:(str(x[0]))))
    # ================== step 6. 输出遗漏的值，并给遗漏的值打上分类标签 ==================
    pre_title = ""
    output_path_txt = "../data/dev_44_pred_EI_omit.txt"
    with open(output_path_txt,'w') as f:
        for title,values in omit.items():
            if pre_title != title: # 如果二者不属于同一篇文章，则换行输出
                for i,line in enumerate(title2doc[title]):
                    f.write(str(i)+" "+" ".join(line)+"\n")
                
            for val in values:
                h_idx,t_idx,relation = val # unpack
                h_entity = title_idx2entity_name[title+str(h_idx)]
                t_entity = title_idx2entity_name[title+str(t_idx)]
                relation_name = relation_id2name[relation_id2label[relation]]
                for _ in out_visual:
                    if (h_entity+", "+t_entity+", "+relation_name) in _:
                        f.write(title+"|"+str(h_idx)+","+str(t_idx)+","+str(relation)+"|"+ _)
            f.write("\n")
            
            pre_title = title

'''
预测的文件结果形式是 h_idx,t_idx 均为数值下标，下面这个函数将下标替换成具体的值
pred_path: 预测的结果文件
golden_path: 原来的文件
'''
def transfer_pred(pred_path,golden_path):
    relation_label2id = get_label2id()
    relation_id2label = dict(zip(relation_label2id.values(), relation_label2id.keys()))
    rel_info_path = '../data/rel_info.json'    
    relation_id2name = get_label_map(rel_info_path) # relation id -> name
    # ================== step 1.根据golden获取数据 ==================
    title_idx2entity_name = {} # 由文本+idx 确定 实体的name
    title_idx2entity_type = {} # 由文本+idx 确定 实体的type
    with open(golden_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont:
            cur_title = dic['title']
            entity_list = dic['vertexSet']
            for idx,entity in enumerate(entity_list):
                title_idx2entity_name[cur_title+str(idx)] = entity[0]['name']
                title_idx2entity_type[cur_title+str(idx)] = entity[0]['type']

    # ================== step 3.找出pred ==================    
    pred_transfer = []
    with open(pred_path,'r') as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic            
            # 拼凑得到一个triplet
            title = dic['title']
            h_idx = dic['h_idx']
            t_idx = dic['t_idx']
            correct = dic['correct'] 
            h_entity = title_idx2entity_name[title+str(h_idx)]
            t_entity = title_idx2entity_name[title+str(t_idx)]
            h_entity_type = title_idx2entity_type[title+str(h_idx)]
            t_entity_type = title_idx2entity_type[title+str(t_idx)]
            score = dic['score']
            relation = relation_id2name[relation_id2label[dic['r_idx']]]
            triplet = (correct,h_entity,h_entity_type,t_entity,t_entity_type,relation,score,title)
            pred_transfer.append(triplet) # 加入一个三元组

    pred_transfer = sorted(pred_transfer,key = lambda x:x[4])
    # ================== step 6. 输出遗漏的值，并给遗漏的值打上分类标签 ==================
    output_path_txt = "../data/dev_pred_transfer.txt"
    with open(output_path_txt,'w') as f:
        for line in pred_transfer:
            f.write(str(line)+"\n")
    return pred_transfer


'''
1.本函数的作用是按照各个类别，并且按照分数从大到小的输出
2.预测的文件结果形式是 h_idx,t_idx 均为数值下标，下面这个函数将下标替换成具体的值
pred_path: 预测的结果文件
golden_path: 原来的文件
'''
def transfer_pred_by_class(pred_path,golden_path,output_path_txt):
    relation_label2id = get_label2id() # {'P1376': 79, ...}
    relation_id2label = dict(zip(relation_label2id.values(), relation_label2id.keys())) #{79: 'P1376',...}
    rel_info_path = '../data/rel_info.json'    
    relation2name = get_label_map(rel_info_path) # {'P6': 'head of government',...}
    id2name = {} # {"1":country,...}
    for key,val in relation_id2label.items():
        id2name[key] = relation2name.get(val,"NA")
    # ================== step 1.根据golden获取数据 ==================
    title_idx2entity_name = {} # 由文本+idx 确定 实体的name
    title_idx2entity_type = {} # 由文本+idx 确定 实体的type
    with open(golden_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont:
            cur_title = dic['title']
            entity_list = dic['vertexSet']
            for idx,entity in enumerate(entity_list):
                title_idx2entity_name[cur_title+str(idx)] = entity[0]['name']
                title_idx2entity_type[cur_title+str(idx)] = entity[0]['type']

    # ================== step 3.找出pred ==================    
    pred_transfer = defaultdict(list)
    with open(pred_path,'r') as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic            
            # 拼凑得到一个triplet
            title = dic['title']
            h_idx = dic['h_idx']
            t_idx = dic['t_idx']
            correct = dic['correct'] 
            # print(title)
            h_entity = title_idx2entity_name[title+str(h_idx)]
            t_entity = title_idx2entity_name[title+str(t_idx)]
            h_entity_type = title_idx2entity_type[title+str(h_idx)]
            t_entity_type = title_idx2entity_type[title+str(t_idx)]
            score = dic['score']
            relation = relation2name[dic['r_idx']]
            index = dic['index']
            triplet = (correct,h_entity,h_entity_type,t_entity,t_entity_type,relation,score,title,index)
            pred_transfer[dic['r_idx']].append(triplet) # 加入一个三元组

    for r in range(1,97):
        pred_transfer[r] = sorted(pred_transfer[r],key = lambda x:x[7],reverse=True)

    # ================== step 6. 输出遗漏的值，并给遗漏的值打上分类标签 ==================
    # output_path_txt = "../data/dev_pred_transfer_by_class.txt"
    with open(output_path_txt,'w') as f:
        for r in range(1,97):
            f.write(id2name[r]+"("+relation_id2label[r]+")\n")
            for line in pred_transfer[relation_id2label[r]]:
                f.write(str(line)+"\n")
            f.write("-"*90+"\n")


# 按照title 分类输出预测结果
def transfer_pred_by_title(pred_path,golden_path,output_path_txt):
    relation_label2id = get_label2id() # {'P1376': 79, ...}
    relation_id2label = dict(zip(relation_label2id.values(), relation_label2id.keys())) #{79: 'P1376',...}
    rel_info_path = '../data/rel_info.json'    
    relation2name = get_label_map(rel_info_path) # {'P6': 'head of government',...}
    id2name = {} # {"1":country,...}
    for key,val in relation_id2label.items():
        id2name[key] = relation2name.get(val,"NA")
    # ================== step 1.根据golden获取数据 ==================
    title_idx2entity_name = {} # 由文本+idx 确定 实体的name
    title_idx2entity_type = {} # 由文本+idx 确定 实体的type
    golden = set()
    with open(golden_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont:
            cur_title = dic['title']
            entity_list = dic['vertexSet']
            labels = dic['labels']
            for idx,entity in enumerate(entity_list):
                title_idx2entity_name[cur_title+str(idx)] = entity[0]['name']
                title_idx2entity_type[cur_title+str(idx)] = entity[0]['type']

            # 将所有的label放到 golden 中
            for label in labels:                
                golden.add((cur_title,label['h'],label['t'],label['r']))

    # ================== step 3.找出pred ==================    
    pred_transfer = []
    with open(pred_path,'r') as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic            
            # 拼凑得到一个triplet
            title = dic['title']
            h_idx = dic['h_idx']
            t_idx = dic['t_idx']
            r_idx = dic['r_idx']
            if "ATLOP" in pred_path:
                correct = (title,h_idx,t_idx,r_idx) in golden            
                score = 1.0
                relation = relation2name[dic['r_idx']]
            elif "EICS" in pred_path:
                correct = dic['correct']
                score = dic['score']
                relation = relation2name[relation_id2label[dic['r_idx']]]
            # print(title)
            h_entity = title_idx2entity_name[title+str(h_idx)]
            t_entity = title_idx2entity_name[title+str(t_idx)]
            h_entity_type = title_idx2entity_type[title+str(h_idx)]
            t_entity_type = title_idx2entity_type[title+str(t_idx)]
            
            triplet = (correct,h_entity,h_entity_type,t_entity,t_entity_type,relation,score,title)
            pred_transfer.append(triplet) # 加入一个三元组

    
    pred_transfer = sorted(pred_transfer,key = lambda x:x[7],reverse=True)

    # ================== step 6. 输出遗漏的值，并给遗漏的值打上分类标签 ==================
    # output_path_txt = "../data/dev_pred_transfer_by_class.txt"
    with open(output_path_txt,'w') as f:
        for line in pred_transfer:        
            f.write(str(line)+"\n")
        


# 统计各个标签类别的数据分布
def realtion_num_statistic(train_path,dev_path):
    label_num_in_train = defaultdict(int)
    label_num_in_dev = defaultdict(int)    
    with open(train_path,'r') as f:
        cont = json.load(f)
        for doc in cont:            
            cur_labels = doc['labels']
            
            for i, label in enumerate(cur_labels):
                label_num_in_train[label['r']] += 1
    
    with open(dev_path,'r') as f:
        cont = json.load(f)
        for doc in cont:            
            cur_labels = doc['labels']
                        
            for i, label in enumerate(cur_labels):                
                label_num_in_dev[label['r']] += 1
    # sorted
    label_num_in_train = sorted(label_num_in_train.items(),key=lambda x:x[1],reverse=True)
    print(label_num_in_train)
    print("--------------")
    label_num_in_dev = sorted(label_num_in_dev.items(),key=lambda x:x[1],reverse=True)
    print(label_num_in_dev)



'''
# 数据增强，主要处理少样本的数据
逻辑：简单的重复少标签的数据
'''
def data_augment_simple(trian_path):
    out_path = "train_aug.json" # 增强后的数据
    with open(train_path,'r') as f:
        cont = json.load(f)
        for doc in cont: 
            cur_labels = doc['labels']
            for i, label in enumerate(cur_labels):                
                label_num_in_dev[label['r']] += 1

    pass

'''
复杂的逻辑，shuffle doc中的句子，需要重新生成label的情况
'''
def data_augment_complex(train_path):
    out_path = "train_600_aug.json" # 增强后的数据
    out = [] 
    with open(train_path,'r') as f:
        cont = json.load(f)        
        for doc in cont:
            new_doc={}
            flag = False
            cur_sents = doc['sents']  # 得到当前的sents
            vertexset = doc['vertexSet']
            labels = doc['labels'] # 得到当前这个doc的labels
            for label in labels:
                if label['r'] in less_label:
                    flag = True
                    break

            if (not flag):                
                continue
            out.append(doc) # 先把之前的存上
            sents_map = [] # 之前的语序
            for i in range(len(cur_sents)):
                sents_map.append((i,cur_sents[i]))
            
            # 对数据进行shuffle
            random.shuffle(sents_map) # 就按照这个位置作为新的sents
            new_sents = []
            old_id2new_id ={} # 老下标 -> 对应新下标
            for new_idx,item in enumerate(sents_map): # 
                old_idx,sent = item
                new_sents.append(sent)
                old_id2new_id[old_idx] = new_idx
            
            # print( [ i['sent_id'] for i in vertexset[0]])
            new_vertexset = copy.deepcopy(vertexset)
            # 对原来的实体的顺序进行修改
            for entity_list in new_vertexset:
                for entity in entity_list:
                    entity['sent_id'] = old_id2new_id[int(entity['sent_id'])]
            
            # print( [ i['sent_id'] for i in vertexset[0]])
            new_doc['vertexSet'] = new_vertexset
            new_doc['labels'] = doc['labels']
            new_doc['title']= doc['title']
            new_doc['sents'] = new_sents
            out.append(new_doc)
    random.shuffle(out) # 随机shuffle一下
    # dump json
    with open(out_path,'w') as f:
        json.dump(out,f)
    

'''
使用数据平衡的方法，降低某些类别的数据
'''
def data_balance(train_path):
    low = 0
    high = 1
    a = np.random.uniform(low,high,1) 
    out_path = "train_600_balance.json" # 平衡后的数据
    out = [] 
    with open(train_path,'r') as f:
        cont = json.load(f)        
        for doc in cont:                        
            vertexset = doc['vertexSet']
            labels = doc['labels'] # 得到当前这个doc的labels
            new_labels = copy.deepcopy(labels)
            for label in labels:
                if label['r'] in more_label:
                    # 随机生成一个数，由此判断是否丢弃该label
                    threshold = np.random.uniform(low,high) 
                    if threshold > 0.5: # 一半的概率丢弃
                        new_labels.remove(label) # 删除这条数据                        

            out.append({"vertexSet":vertexset,"sents":doc['sents'],"labels":new_labels,"title":doc['title']})
            
    # dump json
    with open(out_path,'w') as f:
        json.dump(out,f)


'''
从预测结果中找出不合理的预测结果，主要包括：
1.类型不一致导致的错误 ORG有birthplace
2.
'''


'''
本方法实现功能如下：
1.获取数据集中的所有实体类型=》总共就6种实体类型
2.获取哪些实体间有关系，并给出统计信息
'''
def get_entity_type(path):
    entity_type = []
    triplets_meta = []
    with open(path,'r') as f :
        cont = json.load(f)
        for dic in cont:
            cur_entityid2_type={}
            entity_list = dic['vertexSet']            
            for i in range(len(entity_list)):
                entities = entity_list[i]
                entity_type.append(entities[0]['type'])
                cur_entityid2_type[i] = entities[0]['type']

            # 统计实体（类型+关系+类型）数
            labels = dic['labels']
            for label in labels:
                h = label['h']
                t = label['t']
                r = label['r']
                cur_triplet = (cur_entityid2_type[h],r,cur_entityid2_type[t])
                triplets_meta.append(cur_triplet) # 

    a = Counter(entity_type)
    b = dict(Counter(triplets_meta))
    # b.most_common() # 对整体进行排序
    # b = sorted(b.items(),key = lambda x:x[1],reverse=True)
    print(a)
    # for i in b:
    #     print(i)
    return b



'''
这是用来计算有哪些 （实体类型，关系类型，实体类型）在train.json 中没有出现但是在最后的预测中出现的。
'''
def get_inconsistency(pred_path,golden_path):
    label_map = get_label_map("../data/rel_info.json")
    name_map = dict(zip(label_map.values(), label_map.keys()))

    pred = transfer_pred(pred_path,golden_path)
    triplets = []
    for line in pred:
        triplets.append((line[2],name_map[line[5]],line[4]))
    a = Counter(triplets)
    
    golden_info = get_entity_type("../data/dev.json") # 得到原始的信息
    
    problem_cnt = 0
    problem_key = []
    for item in a.items():
        key,val = item
        if golden_info.get(key) == None:
            problem_cnt+=val
        print(key,"golden=",golden_info.get(key,0),"pred=",val,abs(golden_info.get(key,0)-val)/golden_info.get(key,1))
        if abs(golden_info.get(key,0)-val)/golden_info.get(key,1) > 0.7:
            problem_key.append(key)
    

    for item in golden_info.items():
        key,val = item
        if a.get(key) == None:
            problem_cnt+=val        
    print("有问题的预测个数是：",problem_cnt)

    # print(problem_key)


"""
获取限制数据，限制的情况有：比如 （PER,birth of place，Loc） 这里的LOC 只能有一个
"""
def get_cardinality(pred_path,golden_path):    
    relation_label2id = get_label2id()
    relation_id2label = dict(zip(relation_label2id.values(), relation_label2id.keys()))
    rel_info_path = '../data/rel_info.json'    
    relation_id2name = get_label_map(rel_info_path) # relation id -> name
    
    # ================== step 1.根据golden获取数据 ==================
    title_idx2entity_name = {} # 由文本+idx 确定 实体的name
    title_idx2entity_type = {} # 由文本+idx 确定 实体的type
    with open(golden_path,'r' ) as f:
        cont = json.load(f)
        for dic in cont:
            cur_title = dic['title']
            entity_list = dic['vertexSet']
            for idx,entity in enumerate(entity_list):
                title_idx2entity_name[cur_title+str(idx)] = entity[0]['name']
                title_idx2entity_type[cur_title+str(idx)] = entity[0]['type']

    
    # ================== step 3.找出pred ==================    
    pred_transfer = []
    with open(pred_path,'r') as f:
        cont = json.load(f)
        for dic in cont: # 每个item都是一个dic            
            # 拼凑得到一个triplet
            title = dic['title']
            h_idx = dic['h_idx']
            t_idx = dic['t_idx']
            correct = dic['correct'] 
            h_entity = title_idx2entity_name[title+str(h_idx)]
            t_entity = title_idx2entity_name[title+str(t_idx)]
            h_entity_type = title_idx2entity_type[title+str(h_idx)]
            t_entity_type = title_idx2entity_type[title+str(t_idx)]
            score = dic['score']
            relation = relation_id2name[relation_id2label[dic['r_idx']]]
            triplet = (h_entity,relation,t_entity,h_entity_type,t_entity_type,correct,score,title)
            pred_transfer.append(triplet) # 加入一个三元组

    pred_transfer = sorted(pred_transfer,key = lambda x:x[0]+x[1]+x[2])
    # ================== step 6. 输出遗漏的值，并给遗漏的值打上分类标签 ==================
    output_path_txt = "../data/dev_pred_cardinality.txt"
    with open(output_path_txt,'w') as f:
        for line in pred_transfer:
            f.write(str(line)+"\n")
    return pred_transfer



'''
# subject + relation in unique_relation 对应的object 是唯一的
因为有些subject + relation 只能对应一个object，所以这里找出这些关系的id，然后在后面的处理中仅选择分数最大的一个object
'''
def get_unique_relation_id():
    rel_info_path = '../data/rel_info.json'
    unique_relation = ["father","mother","country of origin","inception","official language"] 
    # 待考虑的关系： "military branch","date of death","start time","end time"] "date of birth","publication date",
    # 加进去没有用的关系： ["place of death","country","place of birth", "country of citizenship","capital","religion","legislative body",]
    print(unique_relation)
    label_map = get_label_map(rel_info_path)
    name2rel = {val:key for key,val in label_map.items()} # 生成
    rel2id = get_label2id()
    unique_relation_id = []
    for _ in unique_relation:
        # print(rel2id[name2rel[_]])
        unique_relation_id.append(rel2id[name2rel[_]])

    return unique_relation_id

"""
将训练数据中的实体 mask 掉。替换的原则是什么?
(1) TODO: 其实替换成 MASK 不一定好，是不是也有别的选择？
['LOC','ORG','PER','MISC','TIME','NUM']
(2) 统一替换成 [MASK]
"""
def mask_entity(train_path_origin,num):
    train_path_augment = "/home/lawson/program/RRCS/data/train_600_mask_20.json"
    res = [] # 增强过的数据集
    with open (train_path_origin,'r') as f:
        cont = json.load(f)
    
    # 做几次循环生成？
    for cnt in range(num):
        # 每篇 doc 
        for doc in cont:
            new_doc = copy.deepcopy(doc) # 得到新的doc
            vertexSet = new_doc['vertexSet']
            labels = new_doc['labels']
            title = new_doc['title']
            sents = new_doc['sents']
            flag = random.uniform(0,1)
            # step 1. 80% 的概率做 mask 操作（扩增数据）
            if flag >= 0: 
                # 待 mask的实体 随机从实体列表中选择
                # step 2. 生成一个随机数值，如果该值大于0.85， 则将其mask成等长度的一个表示
                entity_num = len(vertexSet) 
                for i in range(entity_num):
                    if random.uniform(0,1) >= 0.5: # 对该实体（及其下的所有mention）都进行mask操作。mask成一个逆字符串
                        select_entity  = vertexSet[i]
                        # 对句子/实体内容进行修改
                        for item in select_entity: # 每一项都是mention及其位置信息                        
                            start,end = item['pos'] # 得到开始，结束的位置
                            mention_type = item['type'] 
                            sent_id = item['sent_id'] # 得到需要修改的句子下标
                            cur_name = ''
                            for idx in range(start,end):
                                # 这里的 mention_type 可以对应修改成其它的值
                                sents[sent_id][idx] = '[MASK]' # 替换成对应类型的值
                                cur_name += '['+mention_type+']'
                            item['name'] = cur_name
                         
                # new_doc = {"vertexSet":vertexSet,"labels":labels,"title":title,"sents":sents}
                res.append(doc) # 如果没有修改，则放入之前的doc
                res.append(new_doc) # 放入修改后的doc
            else:
                res.append(doc) # 如果没有修改，则放入之前的doc

    
    # 将最后 augment 得到的数据写入到json 文件中
    with open(train_path_augment,'w') as f:
        json.dump(res,f)


'''
从train.json中统计各个类型的实体数据，想法是：将 [MASK] 替换成对应类型的实体数据，这样可以尽最大可能拟合原文本的分布
(1)
'''
def get_entity():
    train_path = "/home/lawson/program/RRCS/data/train.json"
    with open(train_path,'r') as f:
        cont = json.load(f)
    type_entity = defaultdict(list) #  实体类型到名称的映射表
    for doc in cont:
        entity_list = doc['vertexSet']
        for entity in entity_list:
            for mention in entity:
                cur_type = mention['type']
                type_entity[cur_type].append(mention['name'])
    for i in type_entity.items():
        key,value = i
        type_entity[key] = list(set(value))

    # output_path = "/home/lawson/program/RRCS/data/train_entity_info.json"
    # with open (output_path,'w') as f:
    #     json.dump(type_entity,f)
    return type_entity


'''
根据实体生成对应的token id，并根据其类型和token id长度写到对应的字典中
01.针对的是 train 中的实体
'''
def tokenizer_entity():    
    type_entity = get_entity()
    entity_type_length2token = defaultdict(list) # 由实体类型和长度去得到固定的token 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for item in  type_entity.items():
        key,val = item
        for entity in val:
            tokens = tokenizer.tokenize(entity)
            token_id = tokenizer.convert_tokens_to_ids(tokens)

            # print(a)
            cur_key = key +"_"+ str(len(token_id))
            entity_type_length2token[cur_key].append(token_id)
    return entity_type_length2token



'''
一套模板处理推理的问题，总结需要推理的结果，下面这套模板是在 dev_44.json 的基础上推导出来的。
1.
Washington County, Raleigh Hills, contains administrative territorial entity
Raleigh Hills, Washington County, located in the administrative territorial entity
二者互为推导，有一者均可以推导出另外一者

2. A 的国家是 Greece, A 属于 B集团，所以B集团的国家也是美国 
Skai Group, Greece, country, [0, 1, 4]  （3）需要推导
Skai TV, Skai Group, owned by, 
=> Skai TV, Greece, country

P17:country:1
P127:owned by:34

if ea_1 == eb_2 and eb_1 != ea_2 and r_1 == 1 and r_2 == 34 :
    res.append([ea_2,eb_1,r_1]) # 添加一条新的结果


3. A的国家是 Greece, A 的总部在 B, 所以B的国家也在Greece
Skai TV, Greece, country, [0, 4, 5, 6]   （1）句内分析
Skai TV, Piraeus, headquarters location, [0]  （1）句内分析
=>Piraeus, Greece, country, [0, 4]   （3）需要推导

P17:country:1
P159: headquarters location:29

if ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 1 and r_2 == 29 :
    res.append([eb_2,eb_1,r_1]) 


4. A 位于B中，B的国家是美国，所以A的国家也是美国
West Virginia Route 28, West Virginia, located in the administrative territorial entity, []  （1）句内分析
West Virginia, United States, country, [0]  （1）句内分析
=> West Virginia Route 28, United States, country, [0, 2]  （3）需要推导

P17 : country : 1
P131: "located in the administrative territorial entity : 2

if ea_2 == eb_1 and ea_1 != eb_2 and r_1 == 2 and r_2 == 1:
    res.append([ea_1,eb_2,r_2]) 


5. 
Conrad Oberon Johnson, American, country of citizenship
Conrad Oberon Johnson, Victoria, place of birth
American, Victoria, contains administrative territorial entity


P27: country of citizenship : 3
P19: "place of birth : 14
P150": "contains administrative territorial entity" : 4

if ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 3 and r_2 == 14 :
    res.append([eb_1,eb_2,4]) 

6. 这个区域是在乌兹别克斯坦，所以这个区域的行政中心也在 乌兹别克斯坦
Samarqand Region, Uzbekistan, country
Samarkand, Samarqand Region, capital of
=>Samarkand, Uzbekistan, country

"P1376": "capital of" : 79

if ea_1 == eb_2 and ea_2 != eb_1 and r_1 == 1 and r_2 == 79 :
    res.append([ea_2,eb_1,r_1])


7. 一个人在这里上学，他是美国人，所以这个学校也在美国
Allen Francis Moore, U.S., country of citizenship
Allen Francis Moore, Monticello High School, educated at 
Monticello High School, U.S., country

P27: country of citizenship : 3
"P69": "educated at" : 25

if ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 3 and r_2 == 25:
    res.append([eb_2,eb_1,1])


8. A是B会员，A是C的国民 => B 的国家是 C
Gahn, Swedish, country of citizenship
Gahn, Royal Swedish Academy of Sciences, member of
=> Royal Swedish Academy of Sciences, Swedish, country

"P463": "member of" : 18

if ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 3 and r_2 == 18:
    res.append([eb_2,eb_1,1])


pred_path： 是预测后的结果文件，文件内容示例如下：
{"correct": true, "index": 12, "h_idx": 15, "t_idx": 1, "r_idx": 1, "score": 1.0, "intrain": false, "r": "P17", "title": "Allen County, Ohio"},
post_pro_pred_path : 处理后的文件
threshold: 仅对超过阈值的三元组做推理
'''
def post_processing(pred_path,threshold):
    title2res = defaultdict(list) # 文档名到关系三元组集合        
    with open(pred_path) as f:
        cont = json.load(f)
    
    for item in cont:
        title2res[item['title']].append(item)
    
    res = copy.copy(cont) # 最后的输出结果
    
    for item in title2res.items():        
        title, preds = item #         
        # 双层for 循环，遍历每一条关系，然后找到可以推理出新关系的三元组并写回结果
        for i in range(len(preds)):
            for j in range(len(preds)):
                triplets_a = preds[i]
                ea_1 = triplets_a['h_idx']
                eb_1 = triplets_a['t_idx']
                r_1 = triplets_a['r_idx']

                triplets_b = preds[j]
                ea_2 = triplets_b['h_idx']
                eb_2 = triplets_b['t_idx']
                r_2 = triplets_b['r_idx']                
                
                if ea_1 == eb_2 and eb_1 != ea_2 and r_1 == 1 and r_2 == 34 :
                    # 推导得到的结果要搞个标记
                    tmp = {'correct': "推导", 'h_idx':ea_2, 't_idx': eb_1, 'r_idx': r_1, 'r': 'P17', 'score': 1.0, 'title': title}
                    res.append(tmp) # 添加一条新的结果

                elif ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 1 and r_2 == 29 :
                    tmp = {'correct': "推导", 'h_idx':eb_2, 't_idx': ea_2, 'r_idx': r_1, 'r': 'P17','score': 1.0, 'title': title}
                    res.append(tmp) 

                elif ea_2 == eb_1 and ea_1 != eb_2 and r_1 == 2 and r_2 == 1:
                    tmp = {'correct': "推导", 'h_idx':ea_1, 't_idx': eb_2, 'r_idx': r_2,'r': 'P17', 'score': 1.0, 'title': title}                    
                    res.append(tmp)

                elif ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 3 and r_2 == 14 :
                    tmp = {'correct': "推导", 'h_idx':eb_1, 't_idx': eb_2, 'r_idx': 4,'r':'P150', 'score': 1.0, 'title': title}                    
                    res.append(tmp) 

                elif ea_1 == eb_2 and ea_2 != eb_1 and r_1 == 1 and r_2 == 79 :
                    tmp = {'correct': "推导", 'h_idx':ea_2, 't_idx': eb_1, 'r_idx': r_1, 'r': 'P17','score': 1.0, 'title': title}                    
                    res.append(tmp) 

                elif ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 3 and r_2 == 25:
                    tmp = {'correct': "推导", 'h_idx':eb_2, 't_idx': eb_1, 'r_idx': 1, 'r': 'P17','score': 1.0, 'title': title}                    
                    res.append(tmp) 

                elif ea_1 == ea_2 and eb_1 != eb_2 and r_1 == 3 and r_2 == 18:
                    tmp = {'correct': "推导", 'h_idx':eb_2, 't_idx': eb_1, 'r_idx': 1, 'r': 'P17','score': 1.0, 'title': title}                    
                    res.append(tmp)
    # res = sorted(res,key = lambda x : x['title'])
    out_path = pred_path.strip(".json") + "_template.json"
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(res,f,ensure_ascii=False)


'''
统计 entity_type * entity_type * relation 
train_path: train.json
返回：各个类型的先验概率
'''
def get_type_relation_lable_from_train():
    out_path = "../data/type_realtion_label_train.txt"
    if os.path.exists(out_path):
        return np.loadtxt(out_path,delimiter=',')

    train_path = "../data/dev.json"    
    rel2id_path = '../data/rel2id.json'    
    # 记录 type * type * relation 下的label情况。因为是实体类型，所以就没有了 doc 那个维度，而是直接把整个batch的信息合在一起
    # 把 train.json 下的所有数据都统计出来
    type_relation_label = torch.zeros(6*6, 97).cpu()
    idx_cnt = defaultdict(int) # 用于记录每个type*type 得到的index 的累积值，即当前idx有多少个样本

    with open(train_path,'r') as f:
        cont = json.load(f)
    
    with open(rel2id_path,'r') as f:
        rel2id = json.load(f)

    # 针对每个doc，遍历找出其label，同时找到其 entity_type*relation_type 的映射关系
    for doc in tqdm(cont):
        labels = doc['labels']
        vertexSet = doc['vertexSet'] # 实体
        
        title = doc['title']
        
        # 找出所有的label， 得到所有正样本       
        pos = set()
        for label in labels:
            h_idx = label['h']
            t_idx = label['t']
            r = rel2id[label['r']]
            pos.add((title,h_idx,t_idx,r))

        # 所有样本的组合
        for i in range(len(vertexSet)): # 头实体
            for j in range(len(vertexSet)): # 尾实体
                if i == j:
                    continue
                h_idx_type = vertexSet[i][0]['type']
                t_idx_type = vertexSet[j][0]['type']
                h_type_id = entity_type_name2id[h_idx_type] # 头实体类型
                t_type_id = entity_type_name2id[t_idx_type] # 尾实体类型
                # 因为只有6种实体，所以这里直接乘6
                # [Loc,Loc] => [1,1] => cur_idx = 0
                # [Loc,Num] => [1,6] => cur_idx = 5
                # [Num,Num] => [6,6] => cur_idx = 35
                cur_idx = (h_type_id-1)* 6 + t_type_id - 1
                flag = 0 
                for r in range(1,97):
                    idx_cnt[cur_idx] += 1
                    if (title,i,j,r) in pos: # 如果这种组合在正样本中   
                        # if cur_idx == 0 and r == 3:
                        #     print(title)
                        type_relation_label[cur_idx,r] += 1 # 因为可能不止一个标签，所以要增加1
                        flag = 1

                if flag == 0: # 如果这种组合没有标签，那么直接归为NA类
                    type_relation_label[cur_idx,0] += 1        
    a = type_relation_label.numpy()
    np.savetxt(out_path,a,fmt='%.0f',delimiter=',')
    return a
    # b = get_type_relation_lable_from_pred()
    # c = a-b 
    # cnt = 0
    # for i in range(len(a)):
    #     for j in range(len(b)):
    #         if a[i][j] ==0 and b[i][j] !=0:
    #             cnt +=1 
    # print(cnt)
    # out_path = "../data/type_realtion_label_minus.txt"
    # np.savetxt(out_path,c,fmt='%.0f',delimiter=',')
    # return type_relation_label/torch.sum(type_relation_label,dim=-1,keepdim=True)



'''
从预测结果中，找出entity_type* relation 的关系
'''
def get_type_relation_lable_from_pred():
    dev_path = "../data/dev.json"
    pred_path = "/home/lawson/program/RRCS/code/dev_index_0.61.json"
    rel2id_path = '../data/rel2id.json'    
    
    idx_cnt = defaultdict(int) # 用于记录每个type*type 得到的index 的累积值，即当前idx有多少个样本

    with open(dev_path,'r') as f:
        cont = json.load(f)
    
    with open(rel2id_path,'r') as f:
        rel2id = json.load(f)
    
    entity_id2type_id = {} # 实体id到类型的映射
    pos = set()

    # 针对每个doc，遍历找出其label，同时找到其 entity_type*relation_type 的映射关系
    for doc in tqdm(cont):
        labels = doc['labels']
        vertexSet = doc['vertexSet'] # 实体        
        title = doc['title']

        for i,entity_list in  enumerate(vertexSet):
            key = title+"_"+str(i)
            entity_id2type_id[key] = entity_type_name2id[entity_list[0]['type']]

        for label in labels:
            h_idx = label['h']
            t_idx = label['t']
            r = rel2id[label['r']]            
            pos.add((title,h_idx,t_idx,r))
            # 找出所有的label， 得到所有正样本       
                
    # step2. ==== 查看预测的结果 ====
    with open(pred_path,'r') as f:
        preds = json.load(f)


    # 记录 type * type * relation 下的label情况。因为是实体类型，所以就没有了 doc 那个维度，而是直接把整个batch的信息合在一起
    # 把 train.json 下的所有数据都统计出来
    type_relation_label_pred = torch.zeros(6*6, 97).cpu()
    # 从预测的结果中分析
    for label in preds:
        h_idx = label['h_idx']
        t_idx = label['t_idx']
        r = label['r_idx']
        title = label['title']       

        h_type_id = entity_id2type_id[title+"_"+str(h_idx)] # 头实体类型id
        t_type_id = entity_id2type_id[title+"_"+str(t_idx)] # 尾实体类型id
        
        cur_idx = (h_type_id-1)* 6 + t_type_id - 1                
        type_relation_label_pred[cur_idx,r] += 1 # 因为可能不止一个标签，所以要增加1

    a = type_relation_label_pred.numpy()
    # out_path = "../data/type_realtion_label_pred.txt"
    # np.savetxt(out_path,a,fmt='%.0f',delimiter=',')
    return a


'''
从 doc-level 信息中，随机选取 num 条有关系的单句子出来，作为补充训练
'''
def select_sentece(train_path,num):
    out = []
    out_path = "../data/train_sentence_"+str(num)+".json"
    with open(train_path,'r') as f:
        cont = json.load(f)
    
    # 判断两个有关系的实体是否在同一个句子里面，如果在的话，那么抽出来
    for item in cont:
        labels = item['labels']
        vertexSet = item['vertexSet']
        entity_id2sent_id = defaultdict(list) # entity_id 到 sent_id
        title = item['title']+"_sentence"
        sents = item['sents'] # 获取sents信息

        for i,entity_list in enumerate(vertexSet):
            for entity in entity_list:
                entity_id2sent_id[i].append(entity['sent_id'])


        for label in labels:
            if len(label['evidence']) == 1: # 说明只由句子就可以得到关系
                h_idx = label['h'] # 头实体
                t_idx = label['t'] # 尾实体
                h = vertexSet[h_idx]
                t = vertexSet[t_idx] 
                sent_id = label['evidence'][0]
                h_out = []
                t_out = []
                for i in h:
                    if i['sent_id'] == sent_id:
                        tmp = copy.copy(i)
                        tmp['sent_id'] = 0
                        h_out.append(tmp)
                    
                for i in t:
                    if i['sent_id'] == sent_id:
                        tmp = copy.copy(i)
                        tmp['sent_id'] = 0
                        t_out.append(tmp)
                cur_label = copy.copy(label)

                # 说明这个标注数据有问题，所以不再加入到训练集
                if len(h_out ) == 0 or len(t_out) == 0:
                    continue
                # 重新写头尾实体
                cur_label['h'] = 0
                cur_label['t'] = 1

                cur_doc = {'labels':[cur_label],'sents':[sents[sent_id]],'vertexSet':[h_out,t_out],"title":title}
                out.append(cur_doc) # 加入这条句子，作为后面训练

    random.shuffle(out)
    
    # 将一部分结果写到文件中
    with open(out_path,'w') as f:
        json.dump(out[:num],f)


'''
逻辑简单：删除掉doc 中的某些句子，比如掐头or去尾。将得到的数据作为新doc添加到训练集中
'''
def data_augment(train_path):
    out_path = train_path[:-5]+"_aug_2.json" # 增强后的数据
    out = [] 
    with open(train_path,'r') as f:
        cont = json.load(f)        
        for doc in cont:
            out.append((doc)) # 无论是否修改，都需要保留以前的

            # 删除尾得到新的训练集            
            sents = doc['sents'] 
            sents_idx = [i for i in range(len(sents))] # 生成 sent_id，供后面比较
            vertexset = doc['vertexSet'] # list 
            labels = doc['labels'] # 得到当前这个doc的labels
                        
            # 5%的概率删除最后一条数据，30%的概率删除最后两条，30%的概率删除最后三条，20% 删除最后四条；15% 不变
            # TODO: 10%的概率删除第一条数据，5% 的概率删除前两条数据 暂不实现
            if len(sents) > 4: # 如果句子条数大于4，才执行下面这个策略，否则不执行
                p = random.random()
                if p <= 0.05: # 删除最后一条数据
                    sents_idx.pop()
                    new_sents = sents[:-1] # 去掉最后一句
                elif p>0.1 and p<=0.4: # 删除最后两条数据                    
                    del sents_idx[-2::]
                    new_sents = sents[:-2]
                elif p>0.4 and p<=0.7 : # 删除最后三条
                    del sents_idx[-3::] 
                    new_sents = sents[:-3]
                elif p>=0.8: # 删除最后四条数据
                    del sents_idx[-4::] 
                    new_sents = sents[:-4]
                else: # 其它情况什么都不做
                    # out.append((doc))
                    continue 

            else: # 不做改变
                # out.append((doc))
                continue                        

            new_doc={}
            # 处理entity的信息
            new_vertexset = []
            valid_entity_id = set() # 有效的实体id
            # 如果实体在最后一条句子中，那么直接放弃
            for i,entity_list in enumerate(vertexset):
                entities = []
                for entity in entity_list:
                    if entity['sent_id'] in sents_idx: # 如果在指定的句子中，直接添加
                        entities.append(copy.copy(entity))
                if len(entities) > 0 : # 如果实体在之前就已经存在，那么添加，得到有效的数据集合
                    new_vertexset.append(entities)
                    valid_entity_id.add(i)

            new_labels = []       
            # 删除那些实体 labels 中已经不在 sent_id 的数据
            for label in labels:
                h = label['h']
                t = label['t']
                if h  in valid_entity_id and t in valid_entity_id:
                    new_labels.append(copy.copy(label))
                # else:
                #     print("filter...")
            
            
            # 对数据进行shuffle
            new_doc['vertexSet'] = new_vertexset
            new_doc['labels'] = new_labels
            new_doc['title']= doc['title']
            new_doc['sents'] = new_sents
            out.append(new_doc)


            # 删除头得到新的训练集【第一条句子通常有大量的指代，所以这里暂不处理】

    random.shuffle(out) # 随机shuffle一下
    # dump json
    with open(out_path,'w') as f:
        json.dump(out,f)


# 获取每篇doc下的实体的相对距离，并计算各个距离下的recall、precision、 f1 信息
def get_min_distance_metric(dev_pred_path,dev_path):
    # 统计 golden 中的情况
    with open(dev_path,'r') as f:
        cont = json.load(f)

    title_h_t2dis = {} # 文档title_头实体idx_尾实体_idx => 距离信息        
    
    golden = [ 0 for i in range(1024)] # 记录各个相对距离的 三元组的个数    
    correct = [ 0 for i in range(1024)] # 记录各个相对距离的预测正确的数据    
    all = [0 for i in range(1024)] # 总的预测结果
    golden_set = set() # 存储整个标签信息的集合

    # 遍历doc，找出各个实体间的最短距离
    for doc in cont:
        vertexset = doc['vertexSet'] # list 
        sents = doc['sents']
        title = doc['title'] # doc 的title信息
        labels = doc['labels']
        start = [] # L[i] 表示第i条句子的其实位置，比如L[0] = 0 
        length = 0
        for i,sent in enumerate(sents):
            start.append(length) 
            length += len(sent)

        # 遍历list，找出每个实体的相对距离
        # 计算出每个实体的位置信息，然后放到一个dic中
        entity_idx2loc = defaultdict(list)
        for idx,entity_list in enumerate(vertexset):
            for entity in entity_list:
                loc = start[entity['sent_id']]  + entity['pos'][0] # 得到全局的起始位置
                entity_idx2loc[title+"_"+str(idx)].append(loc)        

        # 计算各个实体间的相对距离        
        rel_dis = [ [0 for h_idx in range(len(vertexset))] for t_idx in range(len(vertexset))]
        for h_idx in range(len(vertexset)):
            for t_idx in range(len(vertexset)):
                h_idx_locs = entity_idx2loc[title+"_"+str(h_idx)]
                t_idx_locs = entity_idx2loc[title+"_"+str(t_idx)]
                min_dis = 1024
                # 找出各个实体的相对位置
                for i in h_idx_locs:
                    for j in t_idx_locs:                        
                        min_dis = min(min_dis,abs(i-j))
                rel_dis[h_idx][t_idx] = min_dis # 这两个实体间的相对距离
        
        title_h_t2dis[title] = rel_dis

        # 遍历label，找出各个距离下的三元组信息
        for label in labels:
            h_idx = label['h']
            t_idx = label['t']
            r_idx = label['r']
            dis = rel_dis[h_idx][t_idx] # 得到两者的相对距离
            # dis2triplets[dis] += 1
            golden[dis] += 1
            golden_set.add((title,h_idx,t_idx,r_idx))
            
    
    with open(dev_pred_path,'r') as f:
        preds = json.load(f)
        for pred in preds:
            title = pred['title'] 
            h_idx = pred['h_idx']
            t_idx = pred['t_idx']
            r = pred['r_idx']
            dis = title_h_t2dis[title][h_idx][t_idx]
            all[dis] += 1
            # if pred.get('correct',0) == 0:
            flag = (title,h_idx,t_idx,r) in golden_set
            # flag = pred['correct']
            if flag : # 如果预测正确
                correct[dis] += 1
    
    # 计算各个位置的recall ,precision ,f1
    # 以10为窗口，计算各个区域内的recall ,f1 ,precision 值
    win_golden = 0
    win_correct = 0
    win_all = 0 
    
    # 统计各个窗口下的值
    recalls = []
    precisions = []
    f1s = []
    windows = []
    goldens = [] 
    for i in range(1,1024):
        if i % 10 == 0: # 开始计算值            
            if win_golden == 0: # 如果真实标签在这个区间就没有值
                continue
            else:
                precision =  win_correct / win_golden

            if win_all == 0: # 没有预测出来
                recall = 0
            else:
                recall = win_correct / win_all            
            
            if recall + precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall+precision)
            
            windows.append(i)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            goldens.append(win_golden)
            # print(f"i = {i},recall = {recall}, precision = {precision}, f1 = {f1}")
            win_all = 0
            win_golden = 0 
            win_correct = 0
        else:
            win_all +=  all[i]
            win_golden += golden[i]
            win_correct += correct[i]
    print(windows)
    print(recalls)
    print(precisions)
    print(f1s)
    print(goldens)
            
if __name__ == '__main__':
    # get_all_entity()    
    train_path = "../data/train.json"
    dev_path = "../data/dev.json"
    test_path = "../data/test.json"
    all_path = "/home/lawson/program/RRCS/data/all.json"
    rel_info_path = '../data/rel_info.json'
    rel2id_path = '../data/rel2id.json'
    dev_44_visual = '../data/dev_44_visual.txt'
    # left_nodes,right_nodes = get_pair_nodes(path,rel_info_path)
    # get_all_entity()
    # analysis_1()
    # predict_by_dict()  
    
    # get_dev_in_train_num()
    # analysis_2()
    # analysis_EPO_num(dev_path)

    # get_pred_omit(pred_path="../data/dev_44_pred_EI.json",golden_path="../data/dev_44.json",dev_44_visual=dev_44_visual)
    # statistic(train_path,dev_path)
    # data_augment(train_path)
    # select_sentece(train_path,800)
    convert_train_data("/home/lawson/program/RRCS/data/redocred/dev_revised_allennlp.json",rel_info_path,out_path="/home/lawson/program/RRCS/data/redocred/dev_revised_allennlp_visual.txt")
    # data_balance(train_path)
    # combine(input_path_1="../data/train.json",input_path_2="../data/train_sentence_800.json",output_path ='../data/train_doc_800_sent.json')
    dev_pred_path = "/home/lawson/program/RRCS/data/dev_pred_EICS.json"
    golden_path = "/home/lawson/program/RRCS/data/dev.json"
    output_path_txt = "/home/lawson/program/RRCS/data/dev_pred_EICS_transfer_by_title.txt"
    # transfer_pred_by_title(dev_pred_path,golden_path,output_path_txt)
    # get_pred_omit(pred_path,golden_path,dev_44_visual)
    # get_entity_type(train_path)
    # get_inconsistency(pred_path,golden_path)
    # get_cardinality(pred_path,golden_path)
    out_path = "../data/train_600_visual.txt"
    # convert_train_data(train_path,rel_info_path,out_path)
    # unique_relation_id = get_cardinality(rel_info_path) # 
    
    # print(get_labelid2name_2())

    # label_map = get_label_map(rel_info_path)
    # a = {val:key for key,val in label_map.items()}
    # # print(a) # name -> relation 
    # name2id = {}
    # label2id= get_label2id() # relation -> id
    # # print(label2id)
    
    # for key,val in a.items():
    #     print("'"+key+"':",0.5,end=",")
    # mask_entity(train_path,1)
    # out_path = "/home/lawson/program/RRCS/data/train_600_mask_20_visual.json"
    # convert_train_data("/home/lawson/program/RRCS/data/train_600_mask_20.json",rel_info_path,out_path)
    # get_entity()
    # tokenizer_entity()
    # analysis_repetitive_entity()
    # for i in (train_path,dev_path):
    #     get_all_entity(i)
    # post_processing("/home/lawson/program/RRCS/data/dev_44_pred.json",0)
    # get_type_relation_lable_from_train()
    # get_labelid2name_2()
    # get_min_distance_metric(dev_pred_path,dev_path)