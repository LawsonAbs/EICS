import json
def process_prediction(path1,path_pred):
    rel2name ={"P6": "head of government", "P17": "country", "P19": "place of birth", "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse", "P27": "country of citizenship", "P30": "continent", "P31": "instance of", "P35": "head of state", "P36": "capital", "P37": "official language", "P39": "position held", "P40": "child", "P50": "author", "P54": "member of sports team", "P57": "director", "P58": "screenwriter", "P69": "educated at", "P86": "composer", "P102": "member of political party", "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher", "P127": "owned by", "P131": "located in the administrative territorial entity", "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains administrative territorial entity", "P155": "follows", "P156": "followed by", "P159": "headquarters location", "P161": "cast member", "P162": "producer", "P166": "award received", "P170": "creator", "P171": "parent taxon", "P172": "ethnic group", "P175": "performer", "P176": "manufacturer", "P178": "developer", "P179": "series", "P190": "sister city", "P194": "legislative body", "P205": "basin country", "P206": "located in or next to body of water", "P241": "military branch", "P264": "record label", "P272": "production company", "P276": "location", "P279": "subclass of", "P355": "subsidiary", "P361": "part of", "P364": "original language of work", "P400": "platform", "P403": "mouth of the watercourse", "P449": "original network", "P463": "member of", "P488": "chairperson", "P495": "country of origin", "P527": "has part", "P551": "residence", "P569": "date of birth", "P570": "date of death", "P571": "inception", "P576": "dissolved, abolished or demolished", "P577": "publication date", "P580": "start time", "P582": "end time", "P585": "point in time", "P607": "conflict", "P674": "characters", "P676": "lyrics by", "P706": "located on terrain feature", "P710": "participant", "P737": "influenced by", "P740": "location of formation", "P749": "parent organization", "P800": "notable work", "P807": "separated from", "P840": "narrative location", "P937": "work location", "P1001": "applies to jurisdiction", "P1056": "product or material produced", "P1198": "unemployment rate", "P1336": "territory claimed by", "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by", "P1376": "capital of", "P1412": "languages spoken, written or signed", "P1441": "present in work", "P3373": "sibling"}
    
    # 得到原始的信息
    # 主要是文章中的实体名等信息
    title_idx2name = {}
    with open(path1,'r') as f:
        cont = json.load(f)
        for doc in cont:
            vertextSet = doc['vertexSet']
            title = doc['title']
            for idx,entity in enumerate(vertextSet):
                title_idx2name[title+str(idx)] = entity[0]['name']
    
    
    out = []
    # 得到预测的结果
    with open(path_pred,'r') as f:
        cont = json.load(f)
        for tmp in cont:            
            title = tmp['title']
            h_idx = tmp['h_idx']
            t_idx = tmp['t_idx']
            r_idx = tmp['r']
            head = title_idx2name[title+str(h_idx)]
            tail = title_idx2name[title+str(t_idx)]
            relation = rel2name[r_idx]
            correct = tmp['correct']
            out.append(f"correct:{correct}, title:{title}, head:{head}, tail:{tail}, relation:{relation}")
    
    out_path = "sdf.txt"
    with open(out_path,'w') as f:
        for line in out:
            f.write(line+"\n")



if __name__ == "__main__":
    path_1 = "/home/lawson/program/RRCS/data/redocred/dev_revised_allennlp.json"
    path_2 = "/home/lawson/program/RRCS/code/logs/20230226/BERT_Rdrop_T_cs_final_allennlp_070333/dev_revised_AREI_pred.json_sort.json"
    process_prediction(path_1,path_2)