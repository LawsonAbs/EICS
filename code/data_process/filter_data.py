import json

# 将带有指示代词的文章挑选出来，得到一个新的数据集，看看这个数据集的大小，同时看看AREI模型在这个数据集上的效果。
def filter_data(path):
    with open(path,'r') as f:
        cont = json.load(f)
    
    res = []
    # 如果该文
    for doc in cont:
        # print(doc)
        vertexSet = doc['vertexSet']
        flag = 0
        for entity in vertexSet:
            for mention in entity:
                if mention['type'] == 'pronoun': # 如果是指示代词
                    flag = 1
                    break
            if flag:
                res.append(doc)
                break
        

    out_file = path+"_filter.json"
    with open(out_file,"w") as f:
        json.dump(res,f)

if __name__ == "__main__":
    filter_data(path = "/home/lawson/program/RRCS/data/redocred/test_revised_allennlp.json")