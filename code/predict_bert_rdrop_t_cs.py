import sklearn.metrics
import torch
from tqdm import tqdm
from config import *
from data_RGCN import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.BERT_rdrop_t_cs import BERT_T,create_heterograph
from utils import get_cuda, logging, print_params,get_all_entity,get_label2id,get_labelid2name
from train_kb import TransR

path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = torch.device("cuda:0")
relid2name = get_labelid2name()

def predict(bert, dataloader,opt, modelname, id2rel, input_theta=-1, output=False, is_test=False, test_prefix='test',
         relation_num=97, ours=False):
    # ours: inter-sentence F1 in LSR
    dev_result = []
    
    for cur_i, d in tqdm(enumerate(dataloader),total=len(dataloader)):    
        with torch.no_grad():
            labels = d['labels'] # 当前这个batch下的数据所有的labels
            L_vertex = d['L_vertex'] # 这个是啥？
            titles = d['titles']
            indexes = d['indexes']
            # TODO 这个参数是什么含义？？
            overlaps = d['overlaps'] 
            # predictions = (batch_size,930,97)
            d['multi_gpu_idx'] = torch.tensor([i for i in range(d['context_idxs'].size(0))]) # 使用这个参数用于控制当前GPU的那批数据的下标
            
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
                                batch_entity_id = d['batch_entity_id'],
                                h_t_pairs_global=d['h_t_pairs_global'],
                                labels = d['labels'] 
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

            j = 0
            # 其实应该直接生成字典遍历，然后匹配predict 即可，这里靠遍历所有的entity下标会导致一个极低的效率
            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        for r in range(1, relation_num):
                            rel_ins = (h_idx, t_idx, r) # 组装成一个rel
                            intrain = label.get(rel_ins, False)  # 判断rel_ins是否在 train 中，如果不在，取False；如果在，取对应值
                            # TODO 下面这个逻辑不懂 => 不重要                            
                            if (ours and (h_idx, t_idx) in overlap) or not ours:
                                dev_result.append((rel_ins in label, float(predict_re[i, j, r]), intrain,title, id2rel[r], index, h_idx, t_idx, r))
                        j += 1  # 对(h_idx,t_idx) 的计数    
    dev_result.sort(key=lambda x: x[1], reverse=True)
    
    # 导出预测结果    
    w = 0
    for i, item in enumerate(dev_result):
        if item[1] >= input_theta:
            w = i

    if output:
        # output = [x[-4:] for x in dev_result[:w+1]]
        res = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1],
                   'score': x[1], 'intrain': x[2],
                   'r': x[-5], 'title': x[-6]} for x in dev_result[:w + 1]]
        json.dump(res, open(test_prefix + "_index.json", "w"))
    


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
        test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)
        test_loader = DGLREDataloader(test_set, batch_size=opt.batch_size, dataset_type='test')
        # transr = TransR(entity_num,relation_num,emb_dim,margin,50)
        bert_t = BERT_T(opt)
        
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
    predict(bert_t, 
            test_loader,
            opt,
            model_name,
            id2rel=id2rel,
            input_theta=opt.input_theta,
            output=True,
            test_prefix='test',
            is_test=True,
            ours=False # TODO ??
            )
    print('predict finished')