#! /bin/bash
# 单纯使用bert 作为baseline，添加focal loss 到其中
export CUDA_VISIBLE_DEVICES=$1

# -------------------GDGN_BERT_base Training Shell Script--------------------

if true; then
  lr=1e-3
  lr_bert=2e-5
  batch_size=16
  test_batch_size=16
  epoch=200
  dev_period=3
  log_step=100
  save_model_freq=2
  negativa_alpha=30
  t_head_num=101
  date=`date +%Y%m%d`
  time=`date +%H%M%S`
  model_name=BERT_BASE
  fig_result_dir=logs/${date}/${model_name}_neg_alpha_${negativa_alpha}_epoch_${epoch}_lr_${lr}_${time}_600
  if [ ! -d ${fig_result_dir} ]; then # 用于判断是否存在文件夹
  mkdir -p ${fig_result_dir}
  notes="仅仅使用Bert的效果"
  fi
  echo ${notes} > /home/lawson/program/RRCS/code/${fig_result_dir}/train.log
  nohup python3 -u train_bert.py \
    --train_set ../data/train_600.json \
    --train_set_save ../data/prepro_data/train_BERT_600.pkl \
    --dev_set ../data/dev_200.json \
    --dev_set_save ../data/prepro_data/dev_BERT_200.pkl \
    --test_set ../data/test.json \
    --test_set_save ../data/prepro_data/test_BERT.pkl \
    --checkpoint_dir ${fig_result_dir}/checkpoint \
    --fig_result_dir ${fig_result_dir} \
    --use_model bert \
    --model_name ${model_name} \
    --lr ${lr} \
    --lr_bert ${lr_bert} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epoch ${epoch} \
    --dev_period ${dev_period} \
    --log_step ${log_step} \
    --save_model_freq ${save_model_freq} \
    --negativa_alpha ${negativa_alpha} \
    --bert_hid_size 768 \
    --bert_path /home/lawson/pretrain/bert-base-uncased \
    --use_entity_type \
    --use_entity_id \
    --dropout 0.6 \
    --activation relu \
    --coslr \
    --fig_file_name ${model_name} \
    >> /home/lawson/program/RRCS/code/${fig_result_dir}/train.log 2>&1 &
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
