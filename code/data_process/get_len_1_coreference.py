import json
# 获取长度为1的指代，其余的指代均去除
def delete_span(path):
    # res = []
    with open(path,'r') as f:
        cont = json.load(f)
        for doc in cont:
            vertexSet = doc['vertexSet']
            new_vertexSet = []
            for entity in vertexSet:
                tmp = []
                for mention in entity:
                    if len(mention['name'].split()) > 1 and mention['type']=='pronoun':
                        # 删除这项
                        continue
                    tmp.append(mention)
                new_vertexSet.append(tmp)
            doc['vertexSet'] = new_vertexSet 


    path = path[0:-6] + "1.json"
    with open(path,'w') as f:
        json.dump(cont,f)

delete_span(path = "/home/lawson/program/RRCS/data/redocred/train_revised_allennlp_2.json")