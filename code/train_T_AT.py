# AT表示采用Adaptive Threshold，同时对loss进行修改
import wandb
import torch.nn.functional as F
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
from data_AT import DGLREDataset,DGLREDataloader, BERTDGLREDataset  # 这是从data.py 中引入几个类
from models.BERT_T_AT import BERT_T_AT
from evaluate_T_AT import eval
from utils import Accuracy, get_cuda, logging, print_params,get_all_entity,get_label2id

matplotlib.use('Agg')

path = "../data/all.json"
entity_num = len(get_all_entity(path))
relation_num = len(get_label2id())
margin = 5
emb_dim = 50 # entity/relation 表示的维度
device = torch.device("cuda:0")
eps=torch.tensor(-1e30,dtype=torch.float32).to(device) # 定义一个极小数，避免出现log(0)，否则训练过程会出现nan
log_softmax = nn.LogSoftmax(dim=-1) # 默认的dim应该是1，所以这里一定要修改一下

# TODO:下面这个损失的实现会在训练负样本的时候，导致损失变为0，大概率是因为exp(x_i)上溢出
# 使用ATLOP中定义的方法计算损失。如下的这个实现过于臃肿，建议还是参考原论文的损失实现
# pos 表示为当前batch中为正样本的下标，neg表示为负样本的下标
# 第0类表示为Threshold class 
# 需要使用relation_mask 去除无关的mask值
def cal_loss_with_adaptive_threshold_bk(predictions,relation_multi_label,relation_mask):    
    zero = torch.tensor(0,dtype=torch.float32).to(torch.device("cuda:0"))
    one = torch.tensor(1,dtype=torch.float32).to(torch.device('cuda:0'))
    eps=1e-20 # 定义一个极小数，避免出现log(0)，否则训练过程会出现nan
    relation_mask = relation_mask.unsqueeze(-1) # 变成3维 [batch_size,max_num,1]
    
    # step1.先找出正样本
    # relation_multi_label 中值为1的就是正样本，反之为负样本。
    # a是分子
    a = torch.where(relation_multi_label>0,torch.exp(predictions),zero) # 如果是正样本，则算出exp{logits}，否则为0
    # 带着Threshold Class，
    # c是分母
    relation_multi_label[:,:,0] = 1 # 第0维的Theshold Class 搞成1
    b = torch.where(relation_multi_label>0,torch.exp(predictions),zero)    
    c = torch.sum(b,dim=-1,keepdim=True) # 求出分母
    # 判断分母是否为0，防止出现 nan 错
    assert torch.sum((a/c +eps)<0) == 0
    d = -torch.log( a/c +eps) # 加上一个小数，避免出现inf 
    relation_multi_label[:,:,0] = 0 # 重置
    d = torch.where(relation_multi_label>0,d,zero) * relation_mask # 过滤掉无关项后，使用relation_mask 过滤掉pad项
    loss_pos = torch.sum(d)/torch.sum(relation_multi_label) # 得到正样本的平均损失

    # step2.再求负样本，求TH类的分数
    neg_relation_multi_label = torch.zeros_like(relation_multi_label) 
    neg_relation_multi_label[:,:,0] = 1 
    e = torch.where(neg_relation_multi_label>0,torch.exp(predictions),zero) # 求分子
    
    # 求出所有负类+Th类的指数和
    temp_relation_multi_label = torch.where(relation_multi_label>0,zero,one) # 每位反过来   
    f = torch.where(temp_relation_multi_label>0,torch.exp(predictions),zero)
    g = torch.sum(f,dim=-1,keepdim=True)
    # assert torch.sum((e/g+eps)<0) == 0
    h = -torch.log(e/g+eps)
    h = torch.where(neg_relation_multi_label>0,h,zero) * relation_mask
    # 因为有padding的情况，所以不能简单的直接除以  relation_multi_label.size(0)*relation_multi_label.size(1)
    # 应该以 relation_mask中1的个数 为基准来计算负样本的个数
    loss_neg = torch.sum(h)/(torch.sum(relation_mask))
    
    return (loss_pos+loss_neg).mean()



# 使用ATLOP中定义的方法计算损失。 为了避免出错，这里使用 log_softmax ，而不是自己手写完整的计算过程
# pos 表示为当前batch中为正样本的下标，neg表示为负样本的下标
# 第0类表示为Threshold class 
# 需要使用relation_mask 去除无关的mask值
# relation_multi_label 中值为1的就是正样本，反之为负样本。    
def cal_loss_with_adaptive_threshold(predictions,relation_multi_label,relation_mask,pos_weight,neg_weight):
    relation_mask = relation_mask.unsqueeze(-1) # 变成3维 [batch_size,max_num,1]    
    # 用于给Threshold class 计算损失的标签位置
    th_label = torch.zeros_like(relation_multi_label) 
    th_label[:,:,0] = 1
    
    relation_multi_label[:,:,0] = 1 # 第0维的Theshold Class 搞成1，即增加TH class
    logits_pos = torch.where(relation_multi_label>0,predictions,eps)  # 得到正样本的预测值

    relation_multi_label[:,:,0] = 0 # 重置
    logits_neg = torch.where(relation_multi_label<=0,predictions,eps)  # 得到负样本的预测值

    # step1.计算正样本的损失
    a = -log_softmax(logits_pos)
    # 拿出对应位置的值，即 relation_multi_label 中为1 的位置
    b = a * relation_multi_label # 按位乘
    loss_pos = torch.sum(b)#/torch.sum(relation_multi_label) # 得到正样本的平均损失
    # loss_pos = torch.sum(b)

    # step2.求负样本求TH类的分数 
    c = -log_softmax(logits_neg)
    d = c * th_label * relation_mask 
        
    # 因为有padding的情况，所以不能简单的直接除以  relation_multi_label.size(0)*relation_multi_label.size(1)
    # 应该以 relation_mask中1的个数 为基准来计算负样本的个数
    loss_neg = torch.sum(d) #/ torch.sum(relation_mask)

    # return (loss_pos + loss_neg ).mean()
    # 使用正负权重调整返回的loss
    return (pos_weight*loss_pos+neg_weight*loss_neg)/torch.sum(relation_mask)


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
        
        model = BERT_T_AT(opt)

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
    BCE = nn.BCEWithLogitsLoss()

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        print('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        lr = chkpt['lr']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    

    model = get_cuda(model) # 将模型放到GPU中
    if opt.use_model == 'bert':        
        bert_param_ids = list(map(id, model.bert.parameters())) # 获取bert模型的参数 
        transformer_param_ids = list(map(id, model.bert_encoder.parameters())) # 获取bert模型的参数 
        # TODO: 验证一下逻辑是否正确
        # 对 model.parameters() 参数进行一个过滤操作
        # 这个操作最后得到的结果不应该是空的吗？ => 不是，会对 线性层等参数进行过滤
        # 返回对象是一个filter
        linar_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids+transformer_param_ids, model.parameters())

        optimizer = optim.AdamW([
            {'params': model.bert.parameters(), 'lr': opt.lr_bert},
            {'params': model.bert_encoder.parameters(), 'lr': opt.lr_tran},
            {'params': linar_params, 'lr': opt.lr},
        ], lr=opt.lr) # 默认 lr

    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                weight_decay=opt.weight_decay)    

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
    
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.title('epoch-f1')
    plt.grid(True)
    # acc_NA 表示的是 负样本的正确率
    # acc_PA 表示的是 正样本的正确率
    # acc_total 
    acc_NA, acc_PA, acc_total = Accuracy(), Accuracy(), Accuracy()
    print('begin..')
    x_epoch = [] # 记录画图时的epoch值
    f1_epoch = [] # 记录f1 值
    precision_epoch = []
    recall_epoch = []

    # 记录每轮epoch结束后，na acc 和 pa acc 的值
    acc_idx = []
    na_epoch = []
    pa_epoch = []
    for epoch in range(start_epoch, opt.epoch + 1):
        start_time = time.time()
        # 每次计算都重新清零
        for acc in [acc_NA, acc_PA, acc_total]:
            acc.clear()

        for ii, d in enumerate(train_loader): # 送入模型的是一个batch 的数据
            relation_multi_label = d['relation_multi_label'] # size = ([batch_size,h_t_limit,97])。 具体含义见data.py定义处声明
            relation_mask = d['relation_mask'] # size = (batch_size,xxx) xxx 表示这个数据是随着 batch 而变化，应该是当前这个batch中具有的最多关系
            # relation_label  和 relation_multi_label 之间的区别是什么？
            # 区别是同一对实体可能有多种关系，relation_multi_label包含了多种关系，relation_label 只随机的取一种关系
            relation_label = d['relation_label'] # size = (batch_size,xxx)
            bsz = len(d['context_idxs']) 
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
            
            # size = (batch_size,930,98) 这个930 是 h_t_pairs 设置得到的
            # predictions[i,j,k] 表示的含义就是 第i个batch，第j对在第k类别的分数 
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
                                h_t_pairs_global=d['h_t_pairs_global'],
                                )                    
            # relation_multi_label 与 predictions 的维度是相同的            
            # 使用 Adaptive threshold 计算损失
            # 计算得到的损失，最后需要除以一个batch size，否则可能会导致在一定epoch之后的梯度爆炸
            loss=cal_loss_with_adaptive_threshold(predictions,relation_multi_label,relation_mask,opt.pos_weight,opt.neg_weight)
            # loss = torch.sum(BCE(predictions, relation_multi_label) * relation_mask.unsqueeze(2)) / (
            #         opt.relation_num * torch.sum(relation_mask)) 
            optimizer.zero_grad()
            loss.backward() # 很是不解，为啥这里需要这个参数？

            if opt.clip != -1:
                nn.utils.clip_grad_value_(model.parameters(), opt.clip)
            optimizer.step()
            # if opt.coslr:  # 原本是在用的
            #     scheduler.step(epoch)

            # relation_label[i][j] 表示第i篇doc，第j个实体对的标签
            # 要使用threshold计算出正负样本的正确率
            relation_label = relation_label.data.cpu().numpy()            
            output = predictions.data.cpu().numpy()
            
            # i 表示第i篇doc
            for i in range(output.shape[0]): # loop 1:得到的是每篇文章中的预测结果
                for j in range(output.shape[1]): # loop 2: entity pair
                    golden_label = relation_label[i][j]
                    if golden_label < 0: # 说明是pad 得到的数据，无需计算
                        break
                    
                    prediction = output[i][j] # 当前这篇doc的所有entity pair预测值
                    cur_threshold = output[i,j,0] # 找出当前 entity pair 的 threshold值

                    idxs = np.argwhere(prediction>cur_threshold).reshape(-1) # 找出所有大于 cur_threshold 的值

                    if len(idxs) == 0: 
                        idx =  0 # 没有预测出任何正样本，则表示当前这条是负样本
                    else: # 为了简单起见，只取一个用于验证
                        idx = idxs[0]


                    # 对预测的结果进行判断                
                    is_correct = (idx == golden_label) 
                    if golden_label == 0:
                        acc_NA.add(is_correct) # 对负样本的预测
                    else:
                        acc_PA.add(is_correct) # 对正样本的预测
                    acc_total.add(is_correct)

            global_step += 1
            # loss = torch.tensor(0)
            # print(loss.item())
            total_loss += loss.item()

            log_step = opt.log_step
            if global_step % log_step == 0:
                cur_loss = total_loss / log_step
                elapsed = time.time() - start_time
                logging(
                    '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.5f} | PA acc: {:4.5f}  | tot acc: {:4.5f} '.format(
                        epoch, global_step, elapsed * 1000 / log_step, cur_loss * 1000, acc_NA.get()*100.0, acc_PA.get()*100.0,
                        acc_total.get()*100.0))
                # 记录 loss 
                wandb.log({"loss": cur_loss,"step":global_step,"epoch":epoch} )
                            
                total_loss = 0
                start_time = time.time()
        
        
        acc_idx.append(epoch)
        na_epoch.append(acc_NA.get()*100.0)
        pa_epoch.append(acc_PA.get()*100.0)

        # 根据 positive acc 来调整验证的频率 => 事实证明，要结合正负样本的正确率来综合判断
        # if acc_PA.get() >= 0.5 and acc_PA.get() < 0.8:
        #     opt.dev_period = 5
        # elif acc_PA.get() >= 0.8: 
        #     opt.dev_period = 1
                        
        if epoch % opt.dev_period == 0 and acc_PA.get() >= 0.5:
            logging('-' * 89)
            eval_start_time = time.time()
            model.eval() # 先设置成eval()
            recall,precision,f1,dev_result = eval(model, dev_loader, model_name, id2rel=id2rel)
            model.train() # 再设置成train()
            logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
            logging('-' * 89)
            
            if f1 > best_ign_f1 :
                best_ign_f1 = f1
                best_epoch = epoch
                path = os.path.join(checkpoint_dir, model_name + '_best.pt')
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    'lr_bert': opt.lr_bert,
                    'lr':opt.lr,
                    'best_epoch': epoch
                }, path)
                dev_result["step"] = global_step # 更新值到字典中
                dev_result["epoch"] = epoch
                wandb.log(dev_result)
                # 记录效果最好时候的 NA acc 和 PA acc
                wandb.log({"NA acc": acc_NA.get()*100.0, "PA acc":acc_PA.get()*100.0,"step":global_step,"epoch":epoch} )

                x_epoch.append(epoch)
                f1_epoch.append(f1)
                precision_epoch.append(precision)
                recall_epoch.append(recall)
    # 所有的epoch跑完之后，开始画图记录所有的效果
    plt.plot(x_epoch,f1_epoch,label="f1")
    plt.plot(x_epoch,recall_epoch,label="recall")
    plt.plot(x_epoch,precision_epoch,label="precision")    
    plt.savefig(os.path.join(fig_result_dir, model_name+"_f1"))

    # 画出acc 的图
    plt.plot(acc_idx,na_epoch,label="na acc")
    plt.plot(acc_idx,pa_epoch,label="pa acc")    
    plt.savefig(os.path.join(fig_result_dir, model_name+"_acc"))

    print("Finish training")
    print("Best epoch = %d | Best Ign F1 = %f" % (best_epoch, best_ign_f1))
    print("Storing best result...")
    print("Finish storing")


if __name__ == '__main__':
    opt = get_opt()  # parser.parse_args() 目的是为了得到输入的参数
    print(opt.notes)
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    
    print("cur file is ",__file__) # 打印当前运行文件的名称
    # TODO 了解一下 __dict__ 属性
    # __dict__ 应该就是打印自己的属性
    print(json.dumps(opt.__dict__, indent=4)) 
    opt.data_word_vec = word2vec  # 这里用的是 config.py中定义的 word2vec。 如果 use_model 使用的是Glove，那么就需要使用到这个参数    
    wandb.init(project="train_bert_T_AT_2", entity="lawsonabs",config=opt)
        
    train(opt)