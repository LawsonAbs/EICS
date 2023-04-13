#! /bin/bash
# -------------------BERT_base Training Shell Script--------------------
export CUDA_VISIBLE_DEVICES=$1

if true; then
  model_name=BERT_Rdrop_T_cs_final_allennlp
  lr=1e-3
  lr_bert=2e-5  
  lr_tran=1e-4
  lr_word_embedding=1e-5
  lr_weight=1e-3
  batch_size=8
  test_batch_size=8
  epoch=70
  dev_period=3
  log_step=100  
  negativa_alpha=-1
  dropout=0.6
  date=`date +%Y%m%d`
  time=`date +%H%M%S`
  bert_dropout=0.22
  pos_lsr=1.0
  neg_lsr=0.65
  interaction_layers=3
  cs_weight=0.07
  entity_type_size=36
  mask_prob=0
  alpha=0
  bert_hid_size=768
  input_theta=0.29019
  fig_result_dir=/home/lawson/program/RRCS/code/logs/${date}/${model_name}_${time}
  if [ ! -d ${fig_result_dir} ]; then # 用于判断是否存在文件夹
  mkdir -p ${fig_result_dir}
  fi
  #进入到指定目录下
  cd /home/lawson/program/RRCS/code
  # 使用指定路径下的python运行
  nohup /home/lawson/anaconda3/bin/python -u evaluate_bert_rdrop_t_cs_allennlp.py \
    --notes "" \
    --train_set ../data/redocred/train_revised_allennlp_1.json \
    --train_set_save ../data/prepro_data/train_revised_allennlp_1_BERT_uncased.pkl \
    --dev_set ../data/redocred/dev_revised_allennlp_1.json \
    --dev_set_save ../data/prepro_data/dev_revised_allennlp_1_BERT_uncased.pkl \
    --test_set ../data/redocred/test_revised_allennlp_1.json \
    --test_set_save ../data/prepro_data/test_revised_allennlp_1_BERT.pkl \
    --pretrain_model "/home/lawson/program/RRCS/code/logs/20230402/BERT_Rdrop_T_cs_final_allennlp_021415/checkpoint/BERT_Rdrop_T_cs_final_allennlp_best.pt" \
    --checkpoint_dir ${fig_result_dir}/checkpoint \
    --input_theta ${input_theta} \
    --fig_result_dir ${fig_result_dir} \
    --bert_path bert-base-uncased \
    --fig_file_name ${model_name} \
    --use_model bert \
    --model_name ${model_name} \
    --lr ${lr} \
    --lr_bert ${lr_bert} \
    --lr_tran ${lr_tran} \
    --lr_weight ${lr_weight} \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --epoch ${epoch} \
    --dev_period ${dev_period} \
    --log_step ${log_step} \
    --negativa_alpha ${negativa_alpha} \
    --bert_hid_size ${bert_hid_size} \
    --dropout ${dropout} \
    --activation relu \
    --coslr \
    --bert_dropout ${bert_dropout} \
    --pos_lsr ${pos_lsr} \
    --neg_lsr ${neg_lsr} \
    --interaction_layers ${interaction_layers} \
    --cs_weight ${cs_weight} \
    --entity_type_size ${entity_type_size} \
    --mask_prob ${mask_prob} \
    --alpha ${alpha} \
    --use_entity_type \
    --use_entity_id \
    >> ${fig_result_dir}/evaluate.log 2>&1 &
fi

