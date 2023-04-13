#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# binary classification threshold, automatically find optimal threshold when -1, default:-1
input_theta=${2--1} # 得到的值是-1
batch_size=5
test_batch_size=16
dataset=test

# -------------------GDGN_BERT_base Evaluation Shell Script--------------------

if true; then
  model_name=GDGN_BERT_base

  nohup python3 -u evaluate.py \
    --train_set ../data/train.json \
    --train_set_save ../data/prepro_data/train_BERT.pkl \
    --dev_set ../data/dev_44.json \
    --dev_set_save ../data/prepro_data/dev_BERT.pkl \
    --test_set ../data/${dataset}.json \
    --test_set_save ../data/prepro_data/${dataset}_BERT.pkl \
    --model_name ${model_name} \
    --use_model bert \
    --pretrain_model checkpoint_4/GDGN_BERT_base_best.pt \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --gcn_dim 808 \
    --gcn_layers 2 \
    --bert_hid_size 768 \
    --bert_path /home/lawson/pretrain/bert-base-uncased \
    --use_entity_type \
    --use_entity_id \
    --dropout 0.6 \
    --activation relu \
    --input_theta ${input_theta} \
    >logs/evaluate_${model_name}_best.log 2>&1 &
fi

# -------------------GDGN_BERT_large Evaluation Shell Script--------------------

if false; then
  model_name=GDGN_BERT_large

  nohup python3 -u test.py \
    --train_set ../data/train.json \
    --train_set_save ../data/prepro_data/train_BERT.pkl \
    --dev_set ../data/dev.json \
    --dev_set_save ../data/prepro_data/dev_BERT.pkl \
    --test_set ../data/${dataset}.json \
    --test_set_save ../data/prepro_data/${dataset}_BERT.pkl \
    --model_name ${model_name} \
    --use_model bert \
    --pretrain_model checkpoint/GDGN_BERT_large_best.pt \
    --batch_size ${batch_size} \
    --test_batch_size ${test_batch_size} \
    --gcn_dim 1064 \
    --gcn_layers 2 \
    --bert_hid_size 1024 \
    --bert_path ../PLM/bert-large-uncased \
    --use_entity_type \
    --use_entity_id \
    --dropout 0.6 \
    --activation relu \
    --input_theta ${input_theta} \
    >logs/test_${model_name}.log 2>&1 &
fi
