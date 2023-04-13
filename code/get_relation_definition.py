import torch as t
from matplotlib.font_manager import json_dump
from tqdm import tqdm
import json 
from transformers import BertTokenizer,BertModel
import requests

'''
获取96种关系的定义，并使用BERT得到它们的表示
'''
class Relation():
    def get_all_relation_name(self): # 获取所有关系的relation名称 （P17 这种）
        with open("/home/lawson/program/RRCS/data/rel2id.json",'r') as f:
            cont = json.load(f)    
        relation_name = []  # [P17,P131,...]
        for line in cont.items():
            relation_name.append(line[0])
        return relation_name

    # 根据 relation_name  得到每个关系在 wididata 中的描述
    def get_relation_description(self,relation_name):
        url = "https://www.wikidata.org/wiki/Special:EntityData/"
        relation_name2des = {}
        for i in tqdm(range(len(relation_name))):
            name = relation_name[i]
            if name == "Na":
                continue
            cur_url = url+name+".json"
            # 请求文件
            res = requests.get(cur_url)
            dic = json.loads(res.text)
            # dic = json.dumps(cont,ensure_ascii=False)
            description = dic['entities'][name]["descriptions"]["en"]["value"]
            relation_name2des[name] = description
        # print(relation_name2des)
        
        output_path = "/home/lawson/program/RRCS/data/relation_name2dec.json"
        # with open(output_path,'w') as f:
        #     json.dumps(relation_name2des,f,encoding='utf-8')

'''
生成relation 的 semantic 表示
01. /home/lawson/program/RRCS/data/relation_name2des.json  这个是relation 的释义信息
output_path = "/home/lawson/program/RRCS/data/relation_name_emb_768.txt"
768维 的那个向量 是直接对desc 取bert的结果

output_path = "/home/lawson/program/RRCS/data/relation_name_emb_1536.txt"
1536维 的那个向量 是直接对desc 取bert的结果，拼接上对 realtion name 取bert的效果

'''
def produce_relation_semantic_embedding():
    with open("/home/lawson/program/RRCS/data/rel_info.json",'r') as f:
        relation_name2name = json.load(f)

    realtion_description_path = "/home/lawson/program/RRCS/data/relation_name2des.json"
    with open(realtion_description_path,'r') as f:
        cont = json.load(f)
    relation_emb = {} # 得到各个关系的语义表示
    bert = BertModel.from_pretrained("bert-base-uncased")
    tokernizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("根据关系定义生成关系的表示....")
    for line in tqdm(cont.items()):
        desc = line[1] # 关系定义
        inputs = tokernizer(desc,return_tensors='pt')
        out = bert(**inputs)
        # step1. 获取关系定义的输出
        relation_desc_emb = out.last_hidden_state[:,0,:].view(-1) # 用CLS向量

        
        # step2.获取关系名的输出【以第0层的输入，即bert中的 word embedding】
        relation_name = relation_name2name[line[0]] # 关系名
        inputs = tokernizer(relation_name,return_tensors='pt')
        out = bert(**inputs,output_hidden_states=True)
        relation_name_emb = out.hidden_states[0][0,1:-1,:] # 去除CLS，SEP的表示
        relation_name_emb = t.mean(relation_name_emb,dim=0) # 按照0维度取均值

        # 将二者关系拼接
        relation_emb[line[0]] = t.cat((relation_name_emb,relation_desc_emb),dim=-1)
    
    print("将生成的表示写入文件....")
    output_path = "/home/lawson/program/RRCS/data/relation_name2emb_1536.txt"
    with open(output_path,'w') as f:
        for line in tqdm(relation_emb.items()):
            f.write(line[0]+"\n")
            f.write(str(line[1].tolist())+"\n")
            # json.dump(relation_emb,f)



'''
1. /home/lawson/program/RRCS/data/relation_name_alias.json 这个是所有relation的 alias 信息。 实际使用时，需要把rel_info.json 信息放在其中。
生成1536
'''
def produce_relation_alias_embedding():
    with open("/home/lawson/program/RRCS/data/rel_info.json",'r') as f:
        relation_name2name = json.load(f)

    realtion_description_path = "/home/lawson/program/RRCS/data/relation_name_alias.json"
    with open(realtion_description_path,'r') as f:
        cont = json.load(f)
    relation_emb = {} # 得到各个关系的语义表示
    bert = BertModel.from_pretrained("bert-base-uncased")
    tokernizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("根据关系定义生成关系的表示....")
    for line in tqdm(cont.items()):
        desc = relation_name2name[line[0]] +";"+ line[1] # 关系名 + 关系alias / 关系定义
        inputs = tokernizer(desc,return_tensors='pt')
        out = bert(**inputs)
        # step1. 获取关系定义的输出
        relation_alias_emb = out.last_hidden_state[:,0,:].view(-1) # 用CLS向量
        relation_emb[line[0]] = relation_alias_emb
        
    print("将生成的表示写入文件....")
    output_path = "/home/lawson/program/RRCS/data/relation_name_alias_768.txt"
    with open(output_path,'w') as f:
        for line in tqdm(relation_emb.items()):
            f.write(line[0]+"\n")
            f.write(str(line[1].tolist())+"\n")
            # json.dump(relation_emb,f)

'''
从文件直接获取 relation 的 semantic 表示
'''
def get_relation_emb():
    path = "/home/lawson/program/RRCS/data/relation_name_alias_768.txt"
    with open(path,'r') as f:
        cont = f.readlines()
    relation_emb = {}
    for line in cont:
        if line.startswith("P"):
            key = line.strip("\n")
        else:
            val = line.strip('[]\n')
            relation_emb[key] = [float(_) for _ in val.split(",")]
    
    # 按照 1-96 这个下标搞一个关系表示
    path = "/home/lawson/program/RRCS/data/rel2id.json"
    with open(path,'r') as f:
        cont = json.load(f)
    id2name = {}
    for key,val in cont.items():
        id2name[val] = key    
    
    res = []
    for i in range(1,97):
        res.append(relation_emb[id2name[i]])

    return res
        

if __name__ == "__main__":
    # relation  = Relation()
    # relation_name = relation.get_all_relation_name()
    # relation.get_relation_description(relation_name)
    # get_relation_emb()
    output_path = "/home/lawson/program/RRCS/data/relation_name_alias_emb_1536.txt"
    # produce_relation_semantic_embedding("/home/lawson/program/RRCS/data/relation_name_alias.json",output_path)
    produce_relation_alias_embedding()