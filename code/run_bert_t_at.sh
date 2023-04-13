#! /bin/bash
# --pretrain_model /home/lawson/program/RRCS/code/checkpoint_20211118/GDGN_BERT_base_30.pt \
# 如何使用多卡训练？
export CUDA_VISIBLE_DEVICES=$1

# -------------------BERT_base Training Shell Script--------------------

if true; then
  model_name=BERT_T_AT
  lr_bert=2e-5
  lr=1e-3
  lr_tran=5e-3
  batch_size=16
  test_batch_size=16
  epoch=50
  dev_period=3
  log_step=50  
  interaction_layers=4
  pos_weight=0.5
  neg_weight=0.5
  dropout=0.6
  bert_dropout=0.22
  date=`date +%Y%m%d`
  time=`date +%H%M%S`
  fig_result_dir=logs/${date}/${model_name}_${time}_epoch_${epoch}_pos_weight_${pos_weight}_lr_${lr}_lr_bert_${lr_bert}_dropout_${dropout}_bert_dropout_${bert_dropout}_interactLayres_${interaction_layers}
  if [ ! -d ${fig_result_dir} ]; then # 用于判断是否存在文件夹
  mkdir -p ${fig_result_dir}
  notes=""
  fi
  echo ${notes} > /home/lawson/program/RRCS/code/${fig_result_dir}/train.log
  nohup python3 -u train_T_AT.py \
    --train_set ../data/train.json \
    --train_set_save ../data/prepro_data/train_BERT_T_AT.pkl \
    --dev_set ../data/dev.json \
    --dev_set_save ../data/prepro_data/dev_BERT_T_AT.pkl \
    --test_set ../data/test.json \
    --test_set_save ../data/prepro_data/test_BERT_T_AT.pkl \
    --checkpoint_dir ${fig_result_dir}/checkpoint \
    --fig_result_dir ${fig_result_dir} \
    --use_model bert \
    --model_name ${model_name} \
    --lr ${lr} \
    --lr_tran ${lr_tran} \
    --lr_bert ${lr_bert} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epoch ${epoch} \
    --dev_period ${dev_period} \
    --log_step ${log_step} \
    --bert_hid_size 768 \
    --bert_path /home/lawson/pretrain/bert-base-uncased \
    --use_entity_type True \
    --use_entity_id True \
    --dropout 0.6 \
    --activation relu \
    --coslr \
    --interaction_layers ${interaction_layers} \
    --pos_weight ${pos_weight} \
    --neg_weight ${neg_weight} \
    >/home/lawson/program/RRCS/code/${fig_result_dir}/train.log 2>&1 &
fi

# -------------------GDGN_BERT_large Training Shell Script--------------------

if false; then
  model_name=GDGN_BERT_large
  lr=0.001
  batch_size=20
  test_batch_size=16
  epoch=300
  dev_period=5
  log_step=20
  save_model_freq=3
  negativa_alpha=4

  nohup python3 -u train.py \
    --train_set ../data/train.json \
    --train_set_save ../data/prepro_data/train_BERT.pkl \
    --dev_set ../data/dev.json \
    --dev_set_save ../data/prepro_data/dev_BERT.pkl \
    --test_set ../data/test.json \
    --test_set_save ../data/prepro_data/test_BERT.pkl \
    --use_model bert \
    --model_name ${model_name} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epoch ${epoch} \
    --dev_period ${dev_period} \
    --log_step ${log_step} \
    --save_model_freq ${save_model_freq} \
    --negativa_alpha ${negativa_alpha} \
    --gcn_dim 1064 \
    --gcn_layers 2 \
    --bert_hid_size 1024 \
    --bert_path ../PLM/bert-large-uncased \
    --use_entity_type \
    --use_entity_id \
    --dropout 0.6 \
    --activation relu \
    --coslr \
    >logs/train_${model_name}.log 2>&1 &
fi

# -------------------additional options--------------------

# option below is used to resume training, it should be add into the shell scripts above
# --pretrain_model checkpoint/GDGN_BERT_base_10.pt \
