from collections import defaultdict,Counter
import wandb
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
from data_T_type_loss import DGLREDataset, DGLREDataloader, BERTDGLREDataset  # 这是从data.py 中引入几个类
from models.BERT_T_type_loss import BERT_T
from evaluate_T_constraint import eval_cons
from utils import Accuracy, get_cuda, logging, print_params,get_all_entity,get_label2id

matplotlib.use('Agg')

path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = torch.device("cuda:0")

# 根据预测值获取得到各个类别关系下的预测概率
# h_t_pairs_entity_type_id 表示当前这个batch中 实体对的类型id
def get_info_from_logits(predictions,threshold,h_t_pairs_entity_type_id,relation_mask):
    # type_type_relation_in_pred =  torch.zeros(7*7+1, 97).cuda(device)    
    predictions = torch.sigmoid(predictions.view(-1,97))  # [total_pair_num,97]
    # TODO: 还需要根据 predictions mask 选择处理
    idxs = (h_t_pairs_entity_type_id[:,:,0] - 1)*7 + h_t_pairs_entity_type_id[:,:,1]
    idxs = idxs.view(-1) # [total_pair_num]
        

    idx_2 = idxs.unsqueeze(-1).expand(-1,50)
    mask = torch.arange(50).unsqueeze(0).expand(predictions.size(0),-1).cuda()
    select_matrix = ((mask==idx_2)+0.0).T # size [50,predictions.size(0)]
    target = select_matrix @ predictions # size [50,97]
    
    # 简单朴素的写法
    # for i in range(len(predictions)):
    #     cur_pred = predictions[i] # 第i对实体的预测值，97 维度
    #     cur_idx = idxs[i].item() # 得到当前预测值对应的下标
    #     if cur_idx > 0:
    #         type_type_relation_in_pred[cur_idx] += cur_pred
    #         idx_cnt[cur_idx] += 1
    
    # 比较target的值与type_type_relation_in_pred 的值是否相同
    # print(torch.sum(target != type_type_relation_in_pred))
    
    idx_cnt = Counter(idxs.tolist())
    # 要取均值
    for item in idx_cnt.items():
        key,val = item
        target[key] /= val

    return target


def train(opt):
    # bert/bilstm 用的数据集不同，一个是BERTDGLDataset，一个是DGLREDataset
    # 同时 model = GDGN_BERT(opt)/GDGN_Glove(opt)  也不相同
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
        
        model = BERT_T(opt)

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

        # model = GDGN_GloVe(opt)
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    # print(model.parameters)
    print_params(model) # 打印模型参数量

    start_epoch = 1
    pretrain_model = opt.pretrain_model    
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1        
        # logging('resume from epoch {} with lr {}'.format(start_epoch, lr))    

    model = get_cuda(model) # 将模型放到GPU中
    if opt.use_model == 'bert':
        # 实验结果证明这个learning rate 对效果的影响很大。几乎相差3个点
        bert_param_ids = list(map(id, model.bert.parameters())) # 将对应的参数变成map 
        transformer_param_ids = list(map(id, model.bert_encoder.parameters())) # 获取bert模型的参数 
        
        # 对 model.parameters() 参数进行一个过滤操作        
        # 返回对象是一个filter
        linar_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids+transformer_param_ids, model.parameters())

        optimizer = optim.AdamW([
            {'params': model.bert.parameters(), 'lr': opt.lr_bert},
            {'params': linar_params, 'weight_decay': opt.weight_decay,'lr':opt.lr},
            {'params': model.bert_encoder.parameters(), 'lr':opt.lr_tran}
        ], )

        # optimizer = optim.RMSprop([
        #     {'params': model.bert.parameters(), 'lr': lr * 0.02},
        #     {'params': base_params, 'weight_decay': opt.weight_decay}
        # ], lr=lr)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                                weight_decay=opt.weight_decay)

    BCE_logits = nn.BCEWithLogitsLoss(reduction='none')
    BCE = nn.BCELoss(reduction='none')

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

    model.train()

    global_step = 0
    total_loss = 0
    total_loss_type = 0
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
        # 每次计算都重新清零
        for acc in [acc_NA, acc_PA, acc_total]:
            acc.clear()

        for ii, d in enumerate(train_loader): # 送入模型的是一个batch 的数据
            bsz = d['relation_mask'].size(0)
            relation_multi_label = d['relation_multi_label'] # size = ([batch_size,h_t_limit,97])。 具体含义见data.py定义处声明
            relation_mask = d['relation_mask'] # size = (batch_size,xxx) xxx 表示这个数据是随着 batch 而变化，应该是当前这个batch中具有的最关系
            relation_label = d['relation_label'] # size = (batch_size,xxx)
            type_relation_label = d['type_relation_label'] 
            h_t_pairs_entity_type_id = d['h_t_pairs_entity_type_id']
            # 验证参数是否更新
            # for param in model.named_parameters():                
            #     name,value = param
            #     print(name)
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
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=relation_mask,
                                batch_entity_id = d['batch_entity_id'],
                                h_t_pairs_global=d['h_t_pairs_global']
                                )
            # print(predictions[0,0:10,0:10])
            # 除以(opt.relation_num * torch.sum(relation_mask)) 是因为想得到平均每个样本的损失
            
            loss_1 = torch.sum(BCE_logits(predictions, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                    opt.relation_num * torch.sum(relation_mask)) 
            
            # 计算type*type*relation 这种不一致导致的损失
            # 首先从 predictions 中找出合理的关系对
            type_type_relation_logits = get_info_from_logits(predictions,opt.threshold,h_t_pairs_entity_type_id,relation_mask)
            # 接着计算loss 
            # TODO: NA 位置是否需要计算loss？ 
            # loss_type = torch.sum(BCE(type_type_relation_logits,type_relation_label))/ (49*97)
            loss_type = torch.sum((type_type_relation_logits-type_relation_label)**2) # MSE
            # loss_type= torch.tensor(0)
            optimizer.zero_grad()
            loss = loss_1 + opt.loss_type_weight*loss_type
            loss.backward() # 很是不解，为啥这里需要这个参数？

            if opt.clip != -1:
                nn.utils.clip_grad_value_(model.parameters(), opt.clip)
            optimizer.step()
            # if opt.coslr:  # 原本是在用的
            #     scheduler.step(epoch)

            output = torch.argmax(predictions, dim=-1)
            output = output.data.cpu().numpy()
            relation_label = relation_label.data.cpu().numpy()

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    label = relation_label[i][j]
                    if label < 0:
                        break

                    is_correct = (output[i][j] == label)
                    if label == 0:
                        acc_NA.add(is_correct)
                    else:
                        acc_PA.add(is_correct)

                    acc_total.add(is_correct)

            global_step += 1
            # loss = torch.tensor(0)
            # print(loss.item())
            total_loss += loss_1.item()
            total_loss_type += loss_type.item()
            log_step = opt.log_step
            if global_step % log_step == 0:
                cur_loss = total_loss / log_step
                cur_loss_type = total_loss_type / log_step
                
                elapsed = time.time() - start_time
                logging(
                    '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} , loss_type {:5.4f} | NA acc: {:4.2f} | PA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                        epoch, global_step, elapsed * 1000 / log_step, cur_loss * 1000,cur_loss_type * 1000, acc_NA.get(), acc_PA.get(),
                        acc_total.get()))
                wandb.log({"loss": cur_loss }, step=global_step)
                total_loss = 0
                total_loss_type = 0
                start_time = time.time()

        # 根据 positive acc 来调整验证的频率
        if acc_PA.get() >= 0.5 and acc_PA.get() < 0.8:
            opt.dev_period = 5
        elif acc_PA.get() >= 0.8 :
            opt.dev_period = 1
        
        # 到了该验证模型的时候
        # TODO: 这里把验证和预测的操作一起完成
        if epoch % opt.dev_period == 0 and acc_PA.get() >=0.5:
            logging('-' * 89)
            eval_start_time = time.time()
            model.eval() # 先设置成eval()
            ign_f1, ign_auc, pr_x, pr_y,dev_result = eval_cons(model, dev_loader, model_name, id2rel=id2rel)
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
                    'best_ign_f1': ign_f1,
                    'best_ign_auc': ign_auc,
                    'best_epoch': epoch
                }, path)
                wandb.log(dev_result, step=global_step)
                plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(fig_result_dir, opt.fig_file_name),dpi=600)

    print("Finish training")
    print("Best epoch = %d | Best Ign F1 = %f" % (best_epoch, best_ign_f1))
    print("Storing best result...")
    print("Finish storing")


if __name__ == '__main__':
    opt = get_opt()  # parser.parse_args() 目的是为了得到输入的参数
    print(opt.notes)
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    
    # TODO 了解一下 __dict__ 属性
    # __dict__ 应该就是打印自己的属性
    print(json.dumps(opt.__dict__, indent=4)) 
    
    opt.data_word_vec = word2vec  # 这里用的是 config.py中定义的 word2vec。 如果 use_model 使用的是Glove，那么就需要使用到这个参数
    wandb.init(project="train_bert_T_type_loss", entity="lawsonabs",config=opt)
    
    # wandb.config.update(opt) # 添加配置信息
    train(opt)