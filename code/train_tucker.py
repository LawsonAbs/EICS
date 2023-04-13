'''
tucker 版的 DLRE
使用tucker 替换RGCN 作为解码器
'''

import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from train_kb import TransR
# 下面会使用到 config 中定义的一些变量和值
from config import *
from data_tucker import DGLREDataset, DGLREDataloader, BERTDGLREDataset  # 这是从data.py 中引入几个类
from models.BERT_TUCKER import  BERT_TUCKER
from evaluate_tucker import eval
from utils import Accuracy, get_cuda, logging, print_params,get_all_entity,get_label2id

matplotlib.use('Agg')
path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度


# for ablation
# from models.RRCS_nomention import RRCS_GloVe, RRCS

def train(opt):
    # bert/bilstm 用的数据集不同，一个是BERTDGLDataset，一个是DGLREDataset
    # 同时 model = RRCS(opt)/RRCS_Glove(opt)  也不相同
    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)
        dev_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                                   instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        # TODO 这里使用了 negative_alpha 参数
        train_loader = DGLREDataloader(train_set, batch_size=opt.batch_size, shuffle=True,
                                       negativa_alpha=opt.negativa_alpha)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.test_batch_size, dataset_type='dev')

        if opt.kb_path is not None:
            transr = TransR(entity_num,relation_num,emb_dim,margin,out_feat=emb_dim)
            transr.load_state_dict(torch.load(opt.kb_path)) 
        else:
            transr = TransR(entity_num,relation_num,emb_dim,margin,out_feat=emb_dim)        
        model = BERT_TUCKER(opt)

    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        dev_set = DGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                               instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        train_loader = DGLREDataloader(train_set, batch_size=opt.batch_size, shuffle=True,
                                       negativa_alpha=opt.negativa_alpha)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.test_batch_size, dataset_type='dev')

        # model = RRCS_GloVe(opt)
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    # print(model.parameters)
    # print_params(model) # 打印模型参数量

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        lr = chkpt['lr']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    else:
        logging('training from scratch with lr {}'.format(lr))
    
    if opt.use_model == 'bert':
        bert_param_ids = list(map(id, model.bert.parameters())) # 将对应的参数变成map 

        # 对 model.parameters() 参数进行一个过滤操作
        # 这个操作最后得到的结果不应该是空的吗？
        # 返回对象是一个filter
        base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())

        optimizer = optim.AdamW([
            {'params': model.bert.parameters(), 'lr': lr * 0.01}, # 这个lr 会变成0.001 * 0.01 = 1e-5
            {'params': base_params, 'weight_decay': opt.weight_decay},
            # {'params':model.tucker.parameters(),'lr':0.0005} # 设置tucker 的参数
        ], lr=lr) 

        # optimizer = optim.RMSprop([
        #     {'params': model.bert.parameters(), 'lr': lr * 0.02},
        #     {'params': base_params, 'weight_decay': opt.weight_decay}
        # ], lr=lr)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                weight_decay=opt.weight_decay)

    BCE = nn.BCEWithLogitsLoss(reduction='none')

    if opt.coslr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.epoch // 4) + 1)

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    fig_result_dir = opt.fig_result_dir
    if not os.path.exists(fig_result_dir):
        os.mkdir(fig_result_dir)

    best_ign_auc = 0.0
    best_ign_f1 = 0.0
    best_epoch = 0
    # model = nn.DataParallel(model) # 分发到多卡上
    model = model.cuda() # 将模型放到GPU中
    model.train()

    global_step = 0
    total_loss = 0

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.title('Precision-Recall')
    plt.grid(True)
    # acc_NA 表示的是 负样本的正确率
    # acc_PA 表示的是 正样本的正确率
    acc_NA, acc_PA, acc_total = Accuracy(), Accuracy(), Accuracy()
    logging('begin..')

    for epoch in range(start_epoch, opt.epoch + 1):
        start_time = time.time()
        # 每次计算（每个 epoch 内）都重新清零
        for acc in [acc_NA, acc_PA, acc_total]:
            acc.clear()

        for ii, d in enumerate(train_loader): # 送入模型的是一个batch 的数据
            bsz = len(d['context_idxs'])
            relation_multi_label = d['relation_multi_label'] # size = ([batch_size,h_t_limit,97])。 具体含义见data.py定义处声明
            relation_mask = d['relation_mask'] # size = (batch_size,xxx) xxx 表示这个数据是随着 batch 而变化，应该是当前这个batch中具有的最关系
            relation_label = d['relation_label'] # size = (batch_size,xxx)            
            # 验证参数是否更新
            # for param in model.named_parameters():                
            #     name,value = param
                # print(name)
                # if "conv.weight" in name:
                #     print("conv.weight",end=",")
                #     print(f"value={value[0,0:10]}")
                
                # if "GCN_layers.1.weight" in name:
                #     print("GCN_layers.1.weight",end=",")
                #     print(f"value={value[0,0,0:10]}")
                
                # if "ent_embedding" in name:
                #     print("ent_embedding",end=",")
                #     print(f"value={value[1,0:10]}")

                # 下面的参数值在训练过程中没有更新
                # if "graph_feat_emb.weight" in name:
                #     print("graph_feat_emb",end=",")
                #     print(f"value={value[0,0:10]}")                
                
            predictions = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],                                
                                entity2mention_table=d['entity2mention_table'],
                                relation_mask=relation_mask,                                                    
                                batch_entity_id = d['batch_entity_id'],
                                )
            # 除以(torch.sum(relation_mask)) 是因为想得到平均每个样本的损失
            # predictions 的size  [bsz,h_t_limit*97] ，
            # relation_multi_lable  的 size [bsz,h_t_limit*97]
            # relation_mask  的 size [bsz,h_t_limit*97]
            loss = torch.sum(BCE(predictions, relation_multi_label) * relation_mask) / (
                    torch.sum(relation_mask))


            # loss = torch.tensor([0],requires_grad=True)
            optimizer.zero_grad()
            loss.backward() 
            if opt.clip != -1:
                nn.utils.clip_grad_value_(model.parameters(), opt.clip)
            optimizer.step()
            # if opt.coslr:  # 原本是在用的
            #     scheduler.step(epoch)

            # 将predictions reshape 
            predictions = predictions.reshape(bsz,-1,97) # reshape 一下，用于后面查找
            output = torch.argmax(predictions, dim=-1)
            output = output.data.cpu().numpy()
            relation_label = relation_label.data.cpu().numpy()

            for i in range(output.shape[0]): # batch size 
                for j in range(output.shape[1]):  # entity pair 
                    golden_label = relation_label[i][j] # 
                    if golden_label < 0: # 如果这个是 -100 ，就直接返回，这个就可以避免计算padding的值
                        break

                    is_correct = (output[i][j] == golden_label)
                    if golden_label == 0:
                        acc_NA.add(is_correct)
                    else:
                        acc_PA.add(is_correct)

                    acc_total.add(is_correct)

            global_step += 1
            total_loss += loss.item()

            log_step = opt.log_step
            if global_step % log_step == 0:
                cur_loss = total_loss / log_step
                elapsed = time.time() - start_time
                logging(
                    '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | PA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                        epoch, global_step, elapsed * 1000 / log_step, cur_loss * 1000, acc_NA.get(), acc_PA.get(),
                        acc_total.get()))
                total_loss = 0
                start_time = time.time()

        # 根据 positive acc 来调整验证的频率
        if acc_PA.get() >= 0.8 :
            opt.dev_period = 10
        elif acc_PA.get() >= 0.9 and acc_PA.get() < 0.95:
            opt.dev_period = 3
        elif acc_PA.get() >= 0.95:
            opt.dev_period = 1


        # 到了该验证模型的时候
        # TODO: 这里把验证和预测的操作一起完成
        if epoch % opt.dev_period == 0 and acc_PA.get()>=0.8:
            logging('-' * 89)
            eval_start_time = time.time()
            model.eval() # 先设置成eval()
            ign_f1, ign_auc, pr_x, pr_y,_ = eval(model, dev_loader, model_name, id2rel=id2rel)
            model.train() # 再设置成train()
            logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
            logging('-' * 89)
            
            if ign_f1 > best_ign_f1:
                best_ign_f1 = ign_f1
                best_ign_auc = ign_auc
                best_epoch = epoch
                path = os.path.join(checkpoint_dir, model_name + '_best.pt')
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    'lr': lr,
                    'best_ign_f1': ign_f1,
                    'best_ign_auc': ign_auc,
                    'best_epoch': epoch
                }, path)

                plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(fig_result_dir, opt.fig_file_name),dpi=600) # dpi 设置保真

        if epoch % opt.save_model_freq == 0 and acc_PA.get()>=0.8:
            # 采用一个字典的形式进行数据存储
            path = os.path.join(checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'checkpoint': model.state_dict(),                
            }, path)

    print("Finish training")
    print("Best epoch = %d | Best Ign F1 = %f" % (best_epoch, best_ign_f1))
    print("Storing best result...")
    print("Finish storing")


if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()  # parser.parse_args() 目的是为了得到输入的参数
    # TODO 了解一下 __dict__ 属性
    # __dict__ 应该就是打印自己的属性
    print(json.dumps(opt.__dict__, indent=4)) 
    opt.data_word_vec = word2vec  # 这里用的是 config.py中定义的 word2vec。 如果 use_model 使用的是Glove，那么就需要使用到这个参数
    train(opt)