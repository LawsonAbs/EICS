import numpy as np
'''
在这个评测函数中，我们区分对待不同的类别
'''
import sklearn.metrics
import torch
from tqdm import tqdm
from config import *
from data_T_type_loss import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.BERT_T import BERT_T
from utils import get_cuda, logging, print_params,get_all_entity,get_label2id,get_unique_relation_id
from train_kb import TransR

path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = torch.device("cuda:0")
unique_relation_id = get_unique_relation_id() # 获取relation_id

"""
根据给出的头尾实体返回这对实体的下标，（从0开始计数）
这个函数应该以字典的形式返回，省的每次调用函数
"""
def get_cur_pair_idx(h_idx,t_idx,length):    
    if h_idx == t_idx:        
        t_idx+=1 # 往后延一位
    flag = 0
    if h_idx < t_idx:    
        flag = 1    
    return h_idx*length+t_idx - h_idx - flag


# for ablation
# from models.GCNRE_nomention import GDGN_GloVe, GDGN_BERT
def eval_cons(model, dataloader, modelname, id2rel, input_theta=-1, output=False, is_test=False, test_prefix='dev',
         relation_num=97, ours=False):
    # ours: inter-sentence F1 in LSR

    total_recall_ignore = 0

    dev_result = []
    total_recall = 0
    
    for cur_i, d in tqdm(enumerate(dataloader),total=len(dataloader)):    
        with torch.no_grad():
            labels = d['labels'] # 当前这个batch下的数据所有的labels，其后的 True/False 代表这个标签是否在train中出现过
            L_vertex = d['L_vertex'] # 这个是啥？
            titles = d['titles']
            indexes = d['indexes']
            # TODO: 这个参数是什么含义？？
            overlaps = d['overlaps'] 
            doc_hentity_tentity2_idx = d['doc_hentity_tentity2_idx']
            # predictions = (batch_size,max_pair_num_in_batch,97)
            predictions = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],                                
                                entity2mention_table=d['entity2mention_table'],
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=None,                                
                                batch_entity_id = d['batch_entity_id'],
                                h_t_pairs_global=d['h_t_pairs_global'],
                                h_t_pairs_entity_type_id=d['h_t_pairs_entity_type_id'],
                                length_mask = d['length_mask']
                                )
            # 之前的做法是直接用sigmoid处理，然后选择大于threshold的值，但是经过分析发现：其实有一部分的数据只需要选择一个即可
            predict_re = torch.sigmoid(predictions) 
                
        # 这个 930 是怎么来的？ => 其实这个930 不是固定的，视batch中的 h_t_limit 大小而定
        predict_re = predict_re.data.cpu().numpy() # size = (batch_size,930,97)
        # 下面这个for循环是为了干什么？ 
        # TODO 下面这个for循环是非常耗时的， 相当于一个 4重for循环（64*20*20*97 = 250w，其实也不是一个很大的循环）。 这里绝对存在一个问题需要优化
        
        # i 表示doc的下标
        # cur_entity_pair_idx 表示实体对数
        for i in range(len(labels)): # labels 是个 list，对应大小为 batch_size
            label = labels[i] # {(3, 1, 4): False, (3, 2, 4): False, ...}
            L = L_vertex[i] # 第i篇doc 的entity数量
            title = titles[i] 
            index = indexes[i]  # 这个有什么用？
            overlap = overlaps[i]
            total_recall += len(label)

            # for l in label.values():
            #     if not l:
            #         total_recall_ignore += 1
                        
            # TODO: 其实应该直接生成字典遍历，然后匹配predict 即可，这里靠遍历所有的entity下标会导致一个极低的效率
            for h_idx in range(L): # 头实体
                for r in range(1, relation_num): # 关系下标
                    # 从唯一性限制中找出那些只需要一个object的配置，然后选择最大的，将剩余的值的predict 全部置为0
                    if r in unique_relation_id:
                        # 因为要遍历整行，所以 t_idx 从 0 开始
                        if h_idx == 0:
                            cur_entity_pair_idx = doc_hentity_tentity2_idx[str(i)+"_"+str(h_idx)+"_"+str(1)]
                        else:
                            cur_entity_pair_idx = doc_hentity_tentity2_idx[str(i)+"_"+str(h_idx)+"_"+str(0)]
                        cur_pred = predict_re[i,cur_entity_pair_idx:cur_entity_pair_idx+L-1,r]
                        max_val = np.amax(cur_pred) # 得到最大值
                        t_idx = np.where(cur_pred ==np.amax(cur_pred))[0][0] # 返回值是一个tuple，所以这里仅取一个
                        
                        if t_idx >= h_idx:
                            t_idx+=1
                        rel_ins = (h_idx, t_idx, r) # 组装成一个rel
                        intrain = label.get(rel_ins, False)
                        dev_result.append((rel_ins in label, float(max_val), intrain,
                                            title, id2rel[r], index, h_idx, t_idx, r))
                        continue
                    # 如果不是唯一性限制关系，那么就要依次遍历所有的实体
                    for t_idx in range(L):
                        if h_idx != t_idx:                            
                            cur_entity_pair_idx = doc_hentity_tentity2_idx[str(i)+"_"+str(h_idx)+"_"+str(t_idx)]
                            rel_ins = (h_idx, t_idx, r) # 组装成一个rel
                            intrain = label.get(rel_ins, False)  # 判断rel_ins是否在 train 中，如果不在，取False；如果在，取对应值
                            # TODO: 下面这个逻辑不懂 => 不重要
                            if (ours and (h_idx, t_idx) in overlap) or not ours:
                                dev_result.append((rel_ins in label, float(predict_re[i, cur_entity_pair_idx, r].item()), intrain,title, id2rel[r], index, h_idx, t_idx, r))
                            
    
    dev_result.sort(key=lambda x: x[1], reverse=True)

    if ours:
        total_recall = 0
        for item in dev_result:
            if item[0]:
                total_recall += 1

    # 计算验证集中所有数据的 f1 情况
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
        if item[1] > input_theta:
            w = i

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    theta = dev_result[f1_pos][1]

    if input_theta == -1:
        w = f1_pos
        input_theta = theta

    auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
    # 这里针对不同情况做输出
    if not is_test: # 不在test
        logging('ALL  : Theta {:3.4f} | F1 {:3.4f},Precision {:3.4f}, Recall {:3.4f}, | AUC {:3.4f}'.format(theta, f1,pr_y[w], pr_x[w], auc))
    else: # 在test
        logging(
            'ma_f1 {:3.4f} | input_theta {:3.4f}, Precision {:3.4f}, Recall {:3.4f}, F1 {:3.4f} | AUC {:3.4f}' \
                .format(f1, input_theta, pr_y[w], pr_x[w], f1_arr[w], auc))

    if output:
        # output = [x[-4:] for x in dev_result[:w+1]]
        output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1],
                   'score': x[1], 'intrain': x[2],
                   'r': x[-5], 'title': x[-6]} for x in dev_result[:w + 1]]
        json.dump(output, open(test_prefix + "_index.json", "w"))


    # 计算 ignore train data 后的验证集效果
    pr_x = []
    pr_y = []
    correct = correct_in_train = 0
    w = 0

    # https://github.com/thunlp/DocRED/issues/47
    for i, item in enumerate(dev_result):
        correct += item[0]
        if item[0] & item[2]:
            correct_in_train += 1
        if correct_in_train == correct:
            p = 0
        else:
            p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
        pr_y.append(p)
        pr_x.append(float(correct) / total_recall)

        if item[1] > input_theta:
            w = i

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    ign_max_f1 = f1_arr.max() # ignore max f1

    auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

    logging(
        'Ignore max_f1 {:3.4f} | inhput_theta {:3.4f} dev_result P {:3.4f} dev_result R {:3.4f} dev_result ignore f1 {:3.4f} | AUC {:3.4f}' \
            .format(ign_max_f1, input_theta, pr_y[w], pr_x[w], f1_arr[w], auc))
    result = {"f1":f1,"ign max F1":ign_max_f1,"ign f1":f1_arr[w],"precision":pr_y[w], "recall":pr_x[w], "input_theta":input_theta}

    return ign_max_f1, auc, pr_x, pr_y , result


if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    opt.data_word_vec = word2vec

    if opt.use_model == 'bert':
        # datasets
        # 这里加载train_set 是为了去除test_set 中的train_set 部分
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)        
        # 添加验证集，进行验证
        dev_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',instance_in_train=train_set.instance_in_train, opt=opt)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.batch_size, dataset_type='dev')
        transr = TransR(entity_num,relation_num,emb_dim,margin,50)
        model = BERT_T(opt)
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
    print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load checkpoint from {}'.format(pretrain_model))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    model = get_cuda(model)
    model.eval()
    print("evaluating...")
    f1, auc, pr_x, pr_y ,dev_theta = eval(model, 
                                        dev_loader,
                                        model_name,
                                        id2rel=id2rel,
                                        input_theta=opt.input_theta,
                                        output=True,
                                        test_prefix='dev',
                                        is_test=True,
                                        ours=False # TODO: ??
                                        )
    print('eval finished')