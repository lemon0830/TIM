<div align="center">
    <img width="25%" alt="TIM" src="https://github.com/">
    <h2>
    TIM: Teaching LM to Translate with Comparison
    </h2>
</div>

# **TIM: Teaching LM to Translate with Comparison**

:star: **Support** :star:
- :LLMs: BLOOM-(e.g., [BLOOM-1b7](https://huggingface.co/bigscience/bloomz-1b7), [BLOOMZ-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt)), LLaMA-(e.g., [LLaMA-7b](),)
- :Data-streaming
- :Distributed training with deepspeed ZeRO stage 1/2/3 

## Quick start

### Environment

We develop TIM with [HuggingFaces's transformers](https://github.com/huggingface/transformers) and [Deepspeed-chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat).

Requirements:
- Python 3.7.9
- Pytorch 1.10.0+cu111
- Transformers 4.29
- accelerate==0.19.0
- numpy==1.22.4
- deepspeed==0.9.0
- scikit-learn

### Datasets

- Training data: train_data/alpaca_reward.json, **[train.data.json](https://huggingface.co/datasets/Lemoooon/Train-for-TIM)**

  Example:
  ```
  Instruction: Translate from Chinese to English.
  
  Input: 国有企业和优势民营企业走进赣南革命老区。
  
  Output: State-owned enterprises and advantageous private enterprises entered the revolutionary base area of south Jiangxi.
  
  Bad Output: enterprises and dominant private visited enterprises  the Gannan base.
  
  ```
  
 - test data: test_data/wmt22, test_data/flores200
 
 ### Instruct Tuning with TIM
 
 We modify the example script in transformers, i.e., `run_clm.py` and `Trainer`; and the code for LoRA in Deepspeed-Chat, i.e., utils.
 We support three training strategies: (1) LoRA, (2) FixEmb, (3) Full
 
 **(1) LoRA: Tuning with Low-rank Matrices**
 
 - sft_reward_training/run_lora.sh
 
 ```
    --only_optimize_lora 
    --lora_dim 8 
    --lora_alpha 16 
    --lora_droppout 0.05 
    --lora_module_name ${LORA_MODULE_NAME} 
 ```

 
