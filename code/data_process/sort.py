import json
# 将得到的文件按照title排序
def sort_res(path):
    with open(path,'r') as f:
        cont = json.load(f)        
        res = sorted(cont,key=lambda x:x['title'])
    
    with open(path+"_sort.json",'w') as f:
        json.dump(res,f)


if __name__ == "__main__":
    sort_res(path = "/home/lawson/program/RRCS/code/logs/20230226/BERT_Rdrop_T_cs_final_allennlp_070333/dev_revised_AREI_pred.json")
