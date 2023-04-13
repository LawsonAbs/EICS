#! /bin/bash
# 后面接了 interaction_layers 层transformer 来做交互，使用BCE作为损失
export CUDA_VISIBLE_DEVICES=$1

# -------------------BERT_base Training Shell Script--------------------

if true; then
  lr=1e-3
  lr_bert=2e-5
  lr_tran=2e-4
  lr_bias=1e-1
  batch_size=16
  test_batch_size=16
  epoch=100
  dev_period=1
  log_step=10
  save_model_freq=1
  negativa_alpha=-1
  interaction_layers=2
  interaction_hidden_dropout_prob=0.1
  t_head_num=8
  window=5
  date=`date +%Y%m%d`
  time=`date +%H%M%S`
  model_name=BERT_BASE_${interaction_layers}T_BCE
  fig_result_dir=logs/${date}/${model_name}_${time}
  if [ ! -d ${fig_result_dir} ]; then # 用于判断是否存在文件夹
  mkdir -p ${fig_result_dir}
  fi
  echo ${notes} > /home/lawson/program/RRCS/code/${fig_result_dir}/train.log
  nohup python3 -u train_T_context.py \
    --note "" \
    --train_set ../data/train_600.json \
    --train_set_save ../data/prepro_data/train_BERT_T_600_uncased_context.pkl \
    --dev_set ../data/dev_200.json \
    --dev_set_save ../data/prepro_data/dev_BERT_T_200_uncased_context.pkl \
    --test_set ../data/test.json \
    --test_set_save ../data/prepro_data/test_BERT_uncased.pkl \
    --checkpoint_dir ${fig_result_dir}/checkpoint \
    --fig_result_dir ${fig_result_dir} \
    --use_model bert \
    --model_name ${model_name} \
    --lr ${lr} \
    --lr_bert ${lr_bert} \
    --lr_tran ${lr_tran} \
    --lr_bias ${lr_bias} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epoch ${epoch} \
    --dev_period ${dev_period} \
    --log_step ${log_step} \
    --save_model_freq ${save_model_freq} \
    --negativa_alpha ${negativa_alpha} \
    --bert_hid_size 768 \
    --bert_path bert-base-uncased \
    --dropout 0.6 \
    --activation relu \
    --coslr \
    --interaction_layers ${interaction_layers} \
    --fig_file_name ${model_name} \
    --t_head_num ${t_head_num} \
    --window ${window} \
    >${fig_result_dir}/train.log 2>&1 &
fi

# -------------------BERT_large Training Shell Script--------------------

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
