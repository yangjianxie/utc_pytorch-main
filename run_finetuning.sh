#!/bin/sh
set -e
CURRENT_DIR=`pwd`
#echo "${CURRENT_DIR}"

# accelerator="gpu"
# strategy="ddp"
# devices=1

train_data_path=/home/yangjx/project1/utc/jd-data-intent/jd_intent_train.txt   #/gemini/data-1/train.txt
test_data_path=/home/yangjx/project1/utc/jd-data-intent/jd_intent_test.txt

config_path=/home/yangjx/project1/utc/rrsong/utc-base/config.json  #/gemini/pretrain/config.json
vocab_path=/home/yangjx/project1/utc/rrsong/utc-base #/gemini/pretrain/

pretrained_model_path=/home/yangjx/project1/utc/rrsong/utc-base/pytorch_model.bin #/gemini/pretrain/pytorch_model.bin
checkpoint_dir=/home/yangjx/project1/utc/rrsong/checkpoint
log_dir=/home/yangjx/project1/utc/rrsong

train_batch_size=16
test_batch_size=16
seq_length=512
learning_rate=1e-5

max_epochs=4

max_grad_norm=1
grad_accum_steps=2
eval_steps=20
logging_steps=20
save_checkpoint_steps=10000

seed=42

echo "Starting the training process..."

fabric run finetune.py \
    --train_data_path $train_data_path \
    --test_data_path $test_data_path \
    --config_path $config_path \
    --vocab_path $vocab_path \
    --pretrained_model_path $pretrained_model_path \
    --checkpoint_dir $checkpoint_dir \
    --log_dir $log_dir \
    --train_batch_size $train_batch_size \
    --test_batch_size $test_batch_size \
    --seq_length $seq_length \
    --learning_rate $learning_rate \
    --max_epochs $max_epochs \
    --max_grad_norm $max_grad_norm \
    --grad_accum_steps $grad_accum_steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --save_checkpoint_steps $save_checkpoint_steps \
    --seed $seed


echo "Training completed ！！！"

