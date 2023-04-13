import sklearn.metrics
import torch
from tqdm import tqdm
from config import *
from data_RGCN_allennlp import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.BERT_rdrop_t_cs_word_embedding_combine_allennlp import BERT_T_CR
from utils import get_cuda, logging, print_params,get_all_entity,get_label2id,get_labelid2name

path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = torch.device("cuda:0")
relid2name = get_labelid2name()

def eval(bert,dataloader,opt, modelname, id2rel, input_theta=-1, output=False, is_test=False, test_prefix='dev',
         relation_num=97, ours=False):
    # ours: inter-sentence F1 in LSR    
    all_labels = [] # 所有的labels
    dev_result = []
    total_recall = 0 # 代表当前验证数据集中所有的gold 标签数
    
    for cur_i, d in tqdm(enumerate(dataloader),total=len(dataloader)):    
        with torch.no_grad():
            labels = d['labels'] # 当前这个batch下的数据所有的labels
            all_labels.extend(labels)
            L_vertex = d['L_vertex'] # 这个是啥？
            titles = d['titles']
            indexes = d['indexes']
            # TODO 这个参数是什么含义？？
            overlaps = d['overlaps'] 
            
            # predictions = (batch_size,max_entity_pair_num,97)
            predictions_bert = bert.forward(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],
                                distance=None,
                                entity2mention_table=d['entity2mention_table'],
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=None,
                                pronoun_id = d['pronoun_id'],
                                entity2pronoun_table=d['entity2pronoun_table'],
                                #capital_mask = d['capital_mask']
                                # batch_entity_id = d['batch_entity_id'],
                                # h_t_pairs_global=d['h_t_pairs_global'],
                                # labels = d['labels'] 
                                )            

            predict_re = torch.sigmoid(predictions_bert)
        # 这个 930 是怎么来的？ => 其实这个930 不是固定的，视batch中的 h_t_limit 大小而定
        predict_re = predict_re.data.cpu().numpy() # size = (batch_size,930,97)
        # 下面这个for循环是为了干什么？ 
        # TODO 下面这个for循环是非常耗时的， 相当于一个 4重for循环（64*20*20*97 = 250w，其实也不是一个很大的循环）。 这里绝对存在一个问题需要优化
        # 存在的问题是：
        for i in range(len(labels)): # labels 是个 list，对应大小为 batch_size
            label = labels[i] # {(3, 1, 4): False, (3, 2, 4): False, ...}
            L = L_vertex[i] # 第i篇doc 的entity数量
            title = titles[i] 
            index = indexes[i]  # 这个有什么用？
            overlap = overlaps[i]
            total_recall += len(label) # 

            # for l in label.values():
            #     if not l:
            #         total_recall_ignore += 1

            j = 0
            # 其实应该直接生成字典遍历，然后匹配predict 即可，这里靠遍历所有的entity下标会导致一个极低的效率
            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        for r in range(1, relation_num):
                            rel_ins = (h_idx, t_idx, r) # 组装成一个rel
                            intrain = label.get(rel_ins, False)  # 判断rel_ins是否在 train 中，如果不在，取False；如果在，取对应值
                            
                            if (ours and (h_idx, t_idx) in overlap) or not ours:
                                dev_result.append((rel_ins in label, float(predict_re[i, j, r]), intrain,
                                                    title, id2rel[r], index, h_idx, t_idx, r))
                        j += 1  # 对(h_idx,t_idx) 的计数
    dev_result.sort(key=lambda x: x[1], reverse=True)

    if ours:
        total_recall = 0
        for item in dev_result:
            if item[0]:
                total_recall += 1

    # case 1. 计算验证集中所有数据的 f1 情况 [正常f1]
    pr_x = []
    pr_y = []
    correct = 0
    w = 0

    if total_recall == 0:
        total_recall = 1

    for i, item in enumerate(dev_result):
        correct += item[0]
        pr_y.append(float(correct) / (i + 1))  # Precision
        pr_x.append(float(correct) / total_recall)  # Recall
        if item[1] >= input_theta: #记录要输出的预测的位置
                w = i
        
    # 得到precision 和 recall  的两个数组，根据这两个数组计算 f1 值
    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20)) # 得到每个位置上的 f1
    max_f1 = f1_arr.max() # 得到当前最大的 f1 值
    f1_pos = f1_arr.argmax() # 找出最大 f1 值对应的下标
    theta = dev_result[f1_pos][1] # 结合 dev_result[pos] 找出 threshold 值

    if input_theta == -1: 
        w = f1_pos  # w 代表的是 取最大f1 时的下标
        input_theta = theta

    # TODO: 根据 precision 和 recall 计算auc
    auc = sklearn.metrics.auc(x=pr_x, y=pr_y) 
    
    # 这里针对不同情况做输出
    if not is_test: # 不在test
        logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, max_f1, auc))
    
    if output:
        # output = [x[-4:] for x in dev_result[:w+1]]
        output = [{'correct':x[0],'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1],
                   'score': x[1], 'intrain': x[2],
                   'r': x[-5], 'title': x[-6]} for x in dev_result[:w + 1]]
        json.dump(output, open(test_prefix + "_pred.json", "w"))


    # case 2. 计算 ignore train data 后的验证集效果
    pr_x = []
    pr_y = []
    correct = correct_in_train = 0
    w = 0

    # https://github.com/thunlp/DocRED/issues/47
    for i, item in enumerate(dev_result):
        correct += item[0]
        if item[0] & item[2]:
            correct_in_train += 1
        if correct_in_train == correct:  # 如果当前预测的准确个数 和 train中的是一模一样的话，则precision直接为0
            p = 0
        else: # 否则计算一下当前这个位置的准确度。 i从0计，所以这里+1。 (i+1 - correct_in_train) 代表总预测个数
            # correct - correct_in_train 代表除去train中预测得到的个数
            p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
        pr_y.append(p) 
        pr_x.append(float(correct) / total_recall) # 放入recall

        if item[1] > input_theta:
            w = i
    
    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    ign_max_f1 = f1_arr.max() # 得到 ignore max f1

    auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

    logging(
        'Ignore f1 {:3.4f} | inhput_theta {:3.4f} Precision {:3.4f} Recall {:3.4f} F1 {:3.4f} | AUC {:3.4f}' \
            .format(ign_max_f1, input_theta, pr_y[w], pr_x[w], f1_arr[w], auc))
    # f1_arr[w] 和 ign_max_f1 区别如下：
    # case a: 计算时没有去除train中的标签数据； case b: 计算时去除了train中的标签数据
    # f1_arr[w] 表示 根据 case a 得到的input_theta，然后得到了下标w，然后返回 ignore f1_arr[w]
    # ign_max_f1 表示根据 case b 中，直接得到的最大的值。
    # 理论上， ign_max_f1 >= f1_arr[w]
    result = {"f1":max_f1,"ign max F1":ign_max_f1,"ign f1":f1_arr[w],"precision":pr_y[w], "recall":pr_x[w], "input_theta":input_theta}


    # case 3. 计算Macro F1 的值
    calculate_macro_f1(dev_result[:w + 1],all_labels)
    return ign_max_f1, auc, pr_x, pr_y ,result


'''
为预测结果计算出macro f1值
pred_result 表示的是当前预测的结果
'''
def calculate_macro_f1(pred_result,labels):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),                    
                    "type": rel_type}
    """
    relation_ids = [i for i in range(97)]
    id2rel = {79: 'P1376', 27: 'P607', 73: 'P136', 63: 'P137', 2: 'P131', 11: 'P527', 38: 'P1412', 33: 'P206', 77: 'P205', 52: 'P449', 34: 'P127', 49: 'P123', 66: 'P86', 85: 'P840', 72: 'P355', 93: 'P737', 84: 'P740', 94: 'P190', 71: 'P576', 68: 'P749', 65: 'P112', 40: 'P118', 1: 'P17', 14: 'P19', 19: 'P3373', 42: 'P6', 44: 'P276', 24: 'P1001', 62: 'P580', 83: 'P582', 64: 'P585', 18: 'P463', 87: 'P676', 46: 'P674', 10: 'P264', 43: 'P108', 17: 'P102', 81: 'P25', 3: 'P27', 26: 'P26', 37: 'P20', 30: 'P22',  95: 'P807', 51: 'P800', 78: 'P279', 88: 'P1336', 5: 'P577', 8: 'P570', 15: 'P571', 36: 'P178', 55: 'P179', 75: 'P272', 35: 'P170', 80: 'P171', 76: 'P172', 6: 'P175', 67: 'P176', 91: 'P39', 21: 'P30', 60: 'P31', 70: 'P36', 58: 'P37', 54: 'P35', 31: 'P400', 61: 'P403', 12: 'P361', 74: 'P364', 7: 'P569', 41: 'P710', 32: 'P1344', 82: 'P488', 59: 'P241', 57: 'P162', 9: 'P161', 47: 'P166', 20: 'P40', 23: 'P1441', 45: 'P156', 39: 'P155', 4: 'P150', 90: 'P551', 56: 'P706', 29: 'P159', 13: 'P495', 53: 'P58', 48: 'P194', 16: 'P54', 28: 'P57', 22: 'P50', 86: 'P1366', 92: 'P1365', 69: 'P937', 50: 'P140', 25: 'P69', 96: 'P1198', 89: 'P1056'}
    rel_id2name = {"P6": "head of government", "P17": "country", "P19": "place of birth", "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse", "P27": "country of citizenship", "P30": "continent", "P31": "instance of", "P35": "head of state", "P36": "capital", "P37": "official language", "P39": "position held", "P40": "child", "P50": "author", "P54": "member of sports team", "P57": "director", "P58": "screenwriter", "P69": "educated at", "P86": "composer", "P102": "member of political party", "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher", "P127": "owned by", "P131": "located in the administrative territorial entity", "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains administrative territorial entity", "P155": "follows", "P156": "followed by", "P159": "headquarters location", "P161": "cast member", "P162": "producer", "P166": "award received", "P170": "creator", "P171": "parent taxon", "P172": "ethnic group", "P175": "performer", "P176": "manufacturer", "P178": "developer", "P179": "series", "P190": "sister city", "P194": "legislative body", "P205": "basin country", "P206": "located in or next to body of water", "P241": "military branch", "P264": "record label", "P272": "production company", "P276": "location", "P279": "subclass of", "P355": "subsidiary", "P361": "part of", "P364": "original language of work", "P400": "platform", "P403": "mouth of the watercourse", "P449": "original network", "P463": "member of", "P488": "chairperson", "P495": "country of origin", "P527": "has part", "P551": "residence", "P569": "date of birth", "P570": "date of death", "P571": "inception", "P576": "dissolved, abolished or demolished", "P577": "publication date", "P580": "start time", "P582": "end time", "P585": "point in time", "P607": "conflict", "P674": "characters", "P676": "lyrics by", "P706": "located on terrain feature", "P710": "participant", "P737": "influenced by", "P740": "location of formation", "P749": "parent organization", "P800": "notable work", "P807": "separated from", "P840": "narrative location", "P937": "work location", "P1001": "applies to jurisdiction", "P1056": "product or material produced", "P1198": "unemployment rate", "P1336": "territory claimed by", "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by", "P1376": "capital of", "P1412": "languages spoken, written or signed", "P1441": "present in work", "P3373": "sibling"}

    # relation_types = [v for v in relation_types if not v == "None"]
    # tp： true positive ; fp: false positive ; total : total gold 
    scores = {rel: {"tp": 0, "fp": 0, "total": 0} for rel in relation_ids + ["ALL"]}
    
    # Count TP, FP and FN per type
    for pred in pred_result:
        correct,cur_relation = pred[0],pred[-1] # 得到当前这个预测的结果        
        if correct: # 预测正确
            scores[cur_relation]['tp'] += 1 
        else:  # 预测错误
            scores[cur_relation]['fp'] += 1

    # 对所有的真实数据进行一个统计
    for label in labels:
        for key,val in label.items():
            
            cur_relation = key[-1] # 最后一个是关系
            scores[cur_relation]['total'] += 1
        

    # Compute per relation Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / scores[rel_type]['total']
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_ids])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_ids])
    total = sum([scores[rel_type]["total"] for rel_type in relation_ids])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / total
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    
    # 去除第0类的影响
    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_ids[1::]])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_ids[1::]])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_ids[1::]])

    
    # 先排序，再输出。针对recall进行排序
    # scores = dict(sorted(scores.items(),key = lambda x:x[1]['r']))
    # scores = dict(sorted(scores.items(),key = lambda x:x[1]['tp']+x[1]['fn'])) # 针对 tp+fn 排序，即小样本排在前面
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))
    
    for item in scores.items():
        key,val = item
        rel_type = key
        if key == "ALL" or key==0:
            continue
        rel_type_name = rel_id2name.get(id2rel[key],id2rel[key])
        # rel_type输出右对齐
        print("\t{:>50}: \tTP: {:>5};\tFP: {:>5};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f}".format(
            rel_type_name,
            scores[rel_type]["tp"],
            scores[rel_type]["fp"],
            # scores[rel_type]["fn"],
            scores[rel_type]["p"],
            scores[rel_type]["r"],
            scores[rel_type]["f1"],
            ))

    return scores, precision, recall, f1

if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    

    if opt.use_model == 'bert':
        # datasets
        # 这里加载train_set 是为了去除test_set 中的train_set 部分
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)        
        # 添加验证集，进行验证
        dev_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',instance_in_train=train_set.instance_in_train, opt=opt)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.batch_size, dataset_type='dev')
        
        # 添加测试集
        test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)
        test_loader = DGLREDataloader(test_set, batch_size=opt.batch_size, dataset_type='test')
                
        bert_t = BERT_T_CR(opt)
    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        dev_set = DGLREDataset(opt.dev_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                                instance_in_train=train_set.instance_in_train, opt=opt)

        dev_loader = DGLREDataloader(dev_set, batch_size=opt.batch_size, dataset_type='dev')
        # model = GDGN_GloVe(opt)
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    import gc

    del train_set
    gc.collect()

    # print(model.parameters)
    # print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model    
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        bert_t.load_state_dict(chkpt['checkpoint_bert'])        
        logging('load checkpoint from {}'.format(pretrain_model))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    bert_t = bert_t.cuda()
    
    bert_t.eval()
    
    print("evaluating...")
    f1, auc, pr_x, pr_y ,dev_theta = eval(bert_t,    
                                        test_loader,
                                        opt,
                                        model_name,
                                        id2rel=id2rel,
                                        # input_theta=opt.input_theta,
                                        output=True,
                                        test_prefix='dev_revised_AREI',
                                        is_test=True,
                                        ours=False # TODO: ??
                                        )
    print('eval finished')