echo ${PATH}
nvidia-smi

ROOT_dir=.
work_dir=${ROOT_dir}/TIM
code_dir=${work_dir}/sft_reward_training
data_dir=${work_dir}/train_data

RATE=0.5
model_name=wmt_bloomz-1b7_sft_reward_mtst3
LOG_FILE=${code_dir}/log.${model_name}

export TRANSFORMERS_CACHE=${data_dir}/cache/
export HF_HOME=${data_dir}/cache/
export TORCH_EXTENSIONS_DIR=${data_dir}/cache/torch_extension/${model_name}
export OMP_NUM_THREADS=20
cd ${code_dir}

save_root_dir=${ROOT_dir}
ckpt_dir=${save_root_dir}/checkpoints/${model_name}

if [ ! -d $ckpt_dir ]; then
    mkdir -p $ckpt_dir
    chmod 777 $ckpt_dir -R
fi

export CXX=g++

GPU=0
premodel=${work_dir}/models/bloomz-7b1-mt

predata=${work_dir}/train.data.json
predatas=${data_dir}/train.data.json

LORA_MODULE_NAME="query_key_value" # for BLOOM
#LORA_MODULE_NAME="q_proj,k_proj,v_proj,o_proj" # for Llama


CUDA_VISIBLE_DEVICES=${GPU} /usr/local/python3/bin/deepspeed \
    ${code_dir}/run_clm.py \
    --model_name_or_path ${premodel} \
    --train_file ${predata} \
    --train_files ${predatas} \
    --streaming \
    --only_optimize_lora \
    --lora_dim 8 \
    --lora_alpha 16 \
    --lora_droppout 0.05 \
    --lora_module_name ${LORA_MODULE_NAME} \
    --rl_alpha ${RATE} \
    --stream_buffer_size 10000 \
    --num_train_epochs 1 \
    --preprocessing_num_workers 10 \
    --ignore_data_skip True \
    --warmup_ratio 0.03 \
    --keep_linebreaks False \
    --logging_steps 100 \
    --save_total_limit 1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --save_steps 500 \
    --max_steps 5000 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --learning_rate 3e-4 \
    --block_size 512 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --deepspeed ${work_dir}/deepspeed_config/ds_config.json \
    --cache_dir ${data_dir}/cache/ \
    --do_train \
    --fp16 \
    --output_dir ${ckpt_dir} \
2>&1 |tee ${LOG_FILE}

wait
