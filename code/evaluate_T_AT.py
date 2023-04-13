'''
使用Threshold Class值进行分类
'''
import sklearn.metrics
import torch
from tqdm import tqdm
from config import *
from data_AT import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.BERT_T_AT import BERT_T_AT
from utils import get_cuda, logging, print_params,get_all_entity,get_label2id
from train_kb import TransR

path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = torch.device("cuda:0")

# for ablation
# from models.GCNRE_nomention import GDGN_GloVe, GDGN_BERT
def eval(model, dataloader, modelname, id2rel, relation_num=97):
    # ours: inter-sentence F1 in LSR    
    dev_result = []
    all_golden = 0 # 计算所有的真实标签个数
    
    for cur_i, d in tqdm(enumerate(dataloader),total=len(dataloader)):    
        with torch.no_grad():
            labels = d['labels'] # 当前这个batch下的数据所有的labels
            L_vertex = d['L_vertex'] # 这个是啥？ => 每篇doc中的实体个数
            titles = d['titles']
            indexes = d['indexes'] # 当前batch中的下标
            # TODO: 这个参数是什么含义？？
            # overlaps = d['overlaps'] 
            # predictions = (batch_size,930,97)
            predictions = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],                                
                                entity2mention_table=d['entity2mention_table'],
                                h_t_pairs=d['h_t_pairs'],                                
                                batch_entity_id = d['batch_entity_id'],
                                h_t_pairs_global=d['h_t_pairs_global']
                                )
            
            predict_re = torch.sigmoid(predictions)
        # 这个 930 是怎么来的？ => 其实这个930 不是固定的，视batch中的 h_t_limit 大小而定
        # predict_re = predict_re.data.cpu().numpy() # size = (batch_size,h_t_limit,98)
        
        
        for i in range(len(labels)): # labels 是个 list，对应大小为 batch_size，其中的内容是每篇doc的label信息
            label = labels[i] # {(3, 1, 4): False, (3, 2, 4): False, ...}
            L = L_vertex[i] # 第i篇doc 的entity数量
            title = titles[i]
            # index = indexes[i]  # 这个有什么用？            
            all_golden += len(label) # 找出对所有的真实数据个数
            j = 0
            
            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        cur_threshold = predict_re[i,j,0].item() # 找出当前 entity pair 的 threshold值
                        topv,topidx = torch.topk(predict_re[i,j],k=4,dim=-1)
                        
                        for value,idx in zip(topv,topidx):
                            if value > cur_threshold:
                                rel_ins = (h_idx, t_idx, idx.item()) # 组装成一个rel
                                # rel_ins in label 表示的是否预测到了label中的数据
                                dev_result.append((rel_ins in label, value.item(), title, id2rel[idx.item()], h_idx, t_idx, idx.item()))
                            else:
                                break
                        j += 1  # 对(h_idx,t_idx)对 的计数
    
    # 计算验证集中所有数据的 f1 情况
    correct = 0
    for i, item in enumerate(dev_result):
        correct += item[0] # 对预测正确的计数
    recall = correct/all_golden
    precision = correct/len(dev_result)
    f1 = 2*recall*precision/(recall+precision)

    # 因为使用了adaptive thresholding，所以就不需要再用排序找dev上最优的threshold了，所以直接计算就可以了
    logging('ALL  : Recall = {:3.4f}, Precision = {:3.4f}, F1 = {:3.4f}'.format(recall,precision,f1))        
    result = {"f1":f1,"precision":precision, "recall":recall}
    
    # TODO: 计算 ignore train data 后的验证集效果
    return recall, precision, f1, result


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
        model = BERT_T_AT(opt)
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
    recall,precision,f1,dev_result = eval(model, 
                                        dev_loader,
                                        model_name,
                                        id2rel=id2rel,
                                        )
    print('eval finished')