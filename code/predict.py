import sklearn.metrics
import torch

from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.GDGN import  GDGN_BERT
from models.GDGN_T import GDGN_BERT_T
from utils import get_cuda, logging, print_params,get_all_entity,get_label2id
from train_kb import  TransR

# for ablation
# from models.GCNRE_nomention import GDGN_GloVe, GDGN_BERT

path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = torch.device("cuda:0")



def predict(model, dataloader, modelname, id2rel, input_theta=-1, output=False, 
         relation_num=97, ours=False):
    # ours: inter-sentence F1 in LSR

    total_recall_ignore = 0

    test_result = []
    total_recall = 0
    total_steps = len(dataloader)
    for cur_i, d in enumerate(dataloader):
        print('step: {}/{}'.format(cur_i, total_steps))

        with torch.no_grad():
            labels = d['labels']
            L_vertex = d['L_vertex']
            titles = d['titles']
            indexes = d['indexes']
            overlaps = d['overlaps']

            predictions = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],
                                distance=None,
                                entity2mention_table=d['entity2mention_table'],
                                mention_graphs=d['graphs'],
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=None,
                                path_table=d['path_table'],
                                entity_graphs=d['entity_graphs'],
                                ht_pair_distance=d['ht_pair_distance'],
                                batch_entity_id = d['batch_entity_id'],
                                h_t_pairs_global=d['h_t_pairs_global'],
                                )

            predict_re = torch.sigmoid(predictions)

        predict_re = predict_re.data.cpu().numpy()

        for i in range(len(labels)):
            label = labels[i]
            L = L_vertex[i]
            title = titles[i]
            index = indexes[i]
            overlap = overlaps[i]
            total_recall += len(label)

            for l in label.values():
                if not l:
                    total_recall_ignore += 1

            j = 0

            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        for r in range(1, relation_num):
                            rel_ins = (h_idx, t_idx, r)
                            intrain = label.get(rel_ins, False)

                            if (ours and (h_idx, t_idx) in overlap) or not ours:
                                test_result.append((rel_ins in label, float(predict_re[i, j, r]), intrain,
                                                    title, id2rel[r], index, h_idx, t_idx, r))

                        j += 1

    test_result.sort(key=lambda x: x[1], reverse=True)

    if output:
        # output = [x[-4:] for x in test_result[:w+1]]
        output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1],
                   'score': x[1], 'intrain': x[2],
                   'r': x[-5], 'title': x[-6]} for x in test_result if x[1] >= input_theta]
        json.dump(output, open("result.json", "w"))
           


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
        test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',
                                    instance_in_train=train_set.instance_in_train, opt=opt)
                
        test_loader = DGLREDataloader(test_set, batch_size=opt.test_batch_size, dataset_type='test')
        
        transr = TransR(entity_num,relation_num,emb_dim,margin,50)
        model = GDGN_BERT_T(opt)
    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        test_set = DGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',
                                instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, batch_size=opt.test_batch_size, dataset_type='test')

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

    predict(model, 
            test_loader,
            model_name,
            id2rel=id2rel,
            input_theta = opt.input_theta,
            output=True,            
            ours=False
            )
                                   
    print('finished')