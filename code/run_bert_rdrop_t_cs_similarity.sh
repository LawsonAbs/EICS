#! /bin/bash
# -------------------GDGN_BERT_base Training Shell Script--------------------
export CUDA_VISIBLE_DEVICES=$1

if true; then
  model_name=BERT_BASE_Rdrop_T_cs_similarity
  lr=1e-3
  lr_bert=2e-5  
  lr_tran=1e-4
  batch_size=8
  test_batch_size=16
  epoch=60
  dev_period=3
  log_step=100  
  negativa_alpha=-1
  dropout=0.6
  date=`date +%Y%m%d`
  time=`date +%H%M%S`
  bert_dropout=0.22
  pos_lsr=1.0
  neg_lsr=0.65
  interaction_layers=2
  cs_weight=0.07
  entity_type_size=36
  fig_result_dir=logs/${date}/${model_name}_${time}
  if [ ! -d ${fig_result_dir} ]; then # 用于判断是否存在文件夹
  mkdir -p ${fig_result_dir}  
  fi  
  nohup python -u train_bert_rdrop_t_cs_similarity.py \
    --notes "使用sim loss, 0.5 " \
    --train_set ../data/train.json \
    --train_set_save ../data/prepro_data/train_BERT_T_uncased.pkl \
    --dev_set ../data/dev.json \
    --dev_set_save ../data/prepro_data/dev_BERT_T_uncased.pkl \
    --test_set ../data/test.json \
    --test_set_save ../data/prepro_data/test_BERT.pkl \
    --checkpoint_dir ${fig_result_dir}/checkpoint \
    --fig_result_dir ${fig_result_dir} \
    --bert_path bert-base-uncased \
    --fig_file_name ${model_name} \
    --use_model bert \
    --model_name ${model_name} \
    --lr ${lr} \
    --lr_bert ${lr_bert} \
    --lr_tran ${lr_tran} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epoch ${epoch} \
    --dev_period ${dev_period} \
    --log_step ${log_step} \
    --negativa_alpha ${negativa_alpha} \
    --bert_hid_size 768 \
    --dropout ${dropout} \
    --activation relu \
    --coslr \
    --bert_dropout ${bert_dropout} \
    --pos_lsr ${pos_lsr} \
    --neg_lsr ${neg_lsr} \
    --interaction_layers ${interaction_layers} \
    --cs_weight ${cs_weight} \
    --entity_type_size ${entity_type_size} \
    >> /home/lawson/program/RRCS/code/${fig_result_dir}/train.log 2>&1 &
fi

# -------------------GDGN_BERT_large Training Shell Script--------------------

