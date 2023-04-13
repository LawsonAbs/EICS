from copy import deepcopy 
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from tqdm import tqdm
import json
pronouns = set(['he','him','his','she','her','it','its','their',
                'He','Him','His','She','Her','It','Its','Their'])

'''
预处理数据
（1）获取每个doc的指示代词信息
'''
def process(file_paths):
    if isinstance(file_paths,list):
        for file_path in file_paths:
            _process(file_path)
    else:
        _process(file_paths)


# 处理单个文件，并写回文件中
def _process(file_path):
    with open(file_path,'r') as f:
        cont = json.load(f)
    for doc in tqdm(cont): # 遍历每个doc
        pron = [] # 存储
        sents = doc['sents']
        for sent_idx,sent in enumerate(sents):
            for word_idx,word in enumerate(sent):
                for target in pronouns:
                    if target == word.lower(): # 如果target 在sent中
                        # 记录idx信息
                        start = word_idx
                        end = word_idx+1
                        pron.append({'sent_id':sent_idx,'pos':[start,end],'name':target})
        doc['pronoun'] = pron # 将指示代词的信息写入到文件中
    
    out_path = file_path[:-5] + "_cr.json" 
    with open(out_path,'w') as f:
        json.dump(cont,f)


# 使用allenNLP 处理doc，将得到的数据写回到数据中
# 处理的结果是把指示代词 + the_名词 这种形式的数据重写到数据中
# 下面这个函数处理起来还是存在一些遗漏的。很多指示代词没能加到标注数据中
def add_pronoun_use_allenNLP():
    # step1: 加载预测模型
    coref_spanbert_large_path = "/home/lawson/pretrain/coref-spanbert-large-2021.03.10.tar.gz"
    predictor = Predictor.from_path(coref_spanbert_large_path)

    
    # 解析生成的结果，并返回新的数据
    def _parse_out(coref_out,raw_doc,text,lens):                    
        new_doc = deepcopy(raw_doc) 
        sentences = new_doc['sents']
        entity_list_doc = new_doc['vertexSet'] # 得到原文给出的实体集合
        entity_list_coref = [] # 通过CR模型得到的entity 集合
        clusters = coref_out['clusters']
        text = text.split()        
        
        for cluster in clusters:# 根据cluster的信息，将下标拆成局部的sent_id + pos 这种
            cr_cluster = []
            for j in cluster:
                left,right = j
                flag = 0

                # 查找一个合适的区间，k代表第k句
                for k in range(len(lens)):
                    if left >= lens[k] and k+1 < len(lens) and left < lens[k+1]: # 判断是否落在第k句中
                        sent_id = k # 第k句                        
                        local_left,local_right = [left-lens[k],right+1-lens[k]]
                        pos = [left-lens[k],right+1-lens[k]]
                        flag = 1
                        break # 不用再往下找了
                # 要确保是找到之后，再把这个位置放入
                if flag:
                    # 只把指示代词放到模型中
                    cur_mention = " ".join(sentences[sent_id][local_left:local_right])
                    cr_cluster.append({"name":cur_mention,'sent_id':k,'pos':pos})
                    print(sentences[k][local_left:local_right])
                # print(text[left:right+1])
            print("\n"+"-"*100)
            entity_list_coref.append(cr_cluster)
        # print("CR后：",entity_list_coref)
        
        # print("\n原doc:",entity_list_doc)
        
        supplement_flag = 0 # 是否补充指代信息
        # 写一个的merge(),用于将上面的两个结果 merge 到一起，从而得到
        # merge的原理就是看：当前出现的这个实体，是否包含指示代词，如果包含指示代词，则融入到模型中。
        for cr_entities in entity_list_coref:
            cr = set()
            for entity in cr_entities: # 找出当前这个cr的所有实体
                cr.add(entity['name'])

            # 找出当前的gold实体集合
            for gold_entities in entity_list_doc:                 
                gold = set() # 每轮集合不同
                for entity in gold_entities:
                    gold.add(entity['name'])

                # 依次判断与哪个gold实体 能否配对
                if pronouns & cr and gold & cr: # 配对成功，则需要merge
                    # 把cr的数据写入到gold中，直接更新即可
                    
                    # 加入时，需要去除重复数据，所以使用for循环操作
                    for item in cr_entities:
                        # len(item['name']) <=2 的原因是allenNLP 会生成过长的结果，所以这里想删除掉
                        if item['name'] not in gold and len(item['name'].split()) <=2:
                            gold_entities.append(item) 
                            supplement_flag = 1
            new_doc['vertexSet'] = entity_list_doc # 修改标注后的数据
        return new_doc,supplement_flag
    

    in_file_path = "/home/lawson/program/RRCS/data/dev.json"
    new_doc = []
    tot_update = 0 
    len_diff = 0
    with open(in_file_path,'r') as f:
        cont = json.load(f)
    
    for doc in tqdm(cont[::-1]):
        sentences = doc['sents']
        # 将sentences 中的所有文本都放到一起
        text = ""
        lens = [0] # 记录每条sentence 第一个字符的起初位置，第一个sentence 起始位置是0
        for sentence in sentences:            
            lens.append(len(sentence) + lens[-1])
            words = " ".join(sentence)
            words += " "            
            # text.strip() # 消除最后的空格 => 这行代码是错误的，strip不是原地修改，所以需要赋值操作
            text += words
        
        
        # 处理当前这篇文章
        # 需要注意：这里的 predict 处理完之后，会得到一个返回的字典。但是这个返回的字典中的document 可能不与原先的document的分词顺序相同（也就是因为分词方法导致的词偏差）例如在文章 Great Bear Lake 中。即 len(text.split()) != coref_out['document']
        coref_out = predictor.predict(document=text)
        # coref_out = {"clusters":[[[0, 3], [40, 40], [74, 75], [87, 87], [98, 99], [153, 154], [171, 173], [177, 178], [186, 187]], [[34, 35], [53, 53], [96, 96]], [[25, 35], [91, 96]], [[10, 11], [103, 117]], [[37, 38], [110, 111], [130, 131], [143, 147]], [[72, 72], [123, 123]], [[80, 81], [169, 169], [191, 192]], [[83, 83], [169, 173]], [[6, 7], [183, 184]]]}

        # 判断 处理后的document 和 之前的 text 是否长度一致？
        if len(coref_out['document']) != len(text.split()):
            len_diff += 1 
            # TODO: 如果二者长度不同，那么可以做一个映射，把coref_out 中的数据修改掉。这个代码逻辑复杂，比较难写。
            def mapping(coref_doc,golden_doc):
                # coref_doc = coref_out['document']
                coref2gold = {} # coreference => 真实的位置
                i = j = 0 # 
                tempA = ""
                
                while(i < len(golden_doc) and j < len(coref_doc)):
                    if golden_doc[i] == coref_doc[j]:            


                        coref2gold[j] = i
                        i += 1
                        j += 1
                    else: # 二者不等， 那肯定是把golden 拆成了 数个 coref_doc 中的项
                        if tempA == golden_doc[i]:
                            tempA = "" # 置空                
                            i+=1
                        else:
                            tempA += coref_doc[j]
                            coref2gold[j] = i
                            j+=1
                return coref2gold
            coref2gold = mapping(coref_out['document'],text.split())
            # 根据得到的coref2gold 再对coref中的数据进行修改
            clusters = coref_out['clusters']
            for cluster in clusters:
                for item in cluster:
                    item[0] = coref2gold[item[0]]
                    item[1] = coref2gold[item[1]]

        # 返回的结果还是一个字典
        cur_res,flag = _parse_out(coref_out,doc,text,lens)
        tot_update += flag
        new_doc.append(cur_res)
               
    print(f"修改前后长度不一致的文档有：{len_diff}")
    print(f"总共修改了{tot_update}篇文档\n")
    out_file_path = in_file_path[:-5]+"_allennlp.json"
    with open(out_file_path,'w') as f:
        json.dump(new_doc,f)
    
# 给没有类型的标注补上 pronoun 类型
def add_pronoun_type(in_file):
    with open(in_file,'r') as f:
        cont = json.load(f)
    
    for doc in cont:
        vertexSet = doc['vertexSet']
        for entity_list in vertexSet: # 
            for entity in entity_list:
                if 'type' not in entity.keys():
                    entity['type'] = 'pronoun' # 表示代词
    # print(cont)

    with open(in_file+"_",'w') as f:
        json.dump(cont,f)

if __name__ =="__main__":   
    # add_pronoun_use_allenNLP()
    add_pronoun_type("/home/lawson/program/RRCS/data/dev_allennlp.json")