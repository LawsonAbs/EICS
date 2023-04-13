'''
去除BERT之后的实验
'''
import sklearn.metrics
import torch
from tqdm import tqdm
from config import *
from data_RGCN import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.rdrop_t_cs import BERT_T
from utils import get_cuda, logging, print_params,get_all_entity,get_label2id,get_labelid2name
from train_kb import TransR

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

    total_recall_ignore = 0

    dev_result = []
    total_recall = 0 # 代表当前验证数据集中所有的gold 标签数
    
    for cur_i, d in tqdm(enumerate(dataloader),total=len(dataloader)):    
        with torch.no_grad():
            labels = d['labels'] # 当前这个batch下的数据所有的labels
            L_vertex = d['L_vertex'] # 这个是啥？
            titles = d['titles']
            indexes = d['indexes']
            # TODO 这个参数是什么含义？？
            overlaps = d['overlaps'] 
            # predictions = (batch_size,930,97)
                        
            predictions_bert,batch_feature_bert = bert.forward(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],
                                distance=None,
                                entity2mention_table=d['entity2mention_table'],
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=None,
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

    return ign_max_f1, auc, pr_x, pr_y ,result


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
        
        # 添加测试集
        test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)
        test_loader = DGLREDataloader(test_set, batch_size=opt.batch_size, dataset_type='test')
        
        transr = TransR(entity_num,relation_num,emb_dim,margin,50)
        bert_t = BERT_T(opt)        
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
                                        input_theta=opt.input_theta,
                                        output=True,
                                        test_prefix='test',
                                        is_test=True,
                                        ours=False # TODO: ??
                                        )
    print('eval finished')