![image](https://github.com/lemon0830/TIM/blob/main/images/Fig_Model.png)

# **TIM: Teaching LM to Translate with Comparison**

:star: **Support** :star:
- LLMs: BLOOM-(e.g., [BLOOM-1b7](https://huggingface.co/bigscience/bloomz-1b7), [BLOOMZ-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt)), LLaMA-(e.g., [LLaMA-7b](https://huggingface.co/yahma/llama-7b-hf),[LLaMA-13b](https://huggingface.co/yahma/llama-13b-hf)), ChatGLM-(e.g., [ChatGLM2-6b](https://huggingface.co/THUDM/chatglm2-6b))
- our Proposed TIM [[run_clm.py](https://github.com/lemon0830/TIM/blob/main/sft_reward_training/run_clm.py)] and Vanilla Instruct-tuning[[run_clm_sft.py]](https://github.com/lemon0830/TIM/blob/main/sft_reward_training/run_clm_sft.py)
- LoRA, Tuning with Embedding Fixed, Full Parameters Tuning
- [Data-streaming](https://github.com/huggingface/datasets/blob/5f810b7011a8a4ab077a1847c024d2d9e267b065/docs/source/stream.mdx)
- Distributed training with [deepspeed ZeRO stage 1/2/3](https://huggingface.co/docs/transformers/main_classes/deepspeed) 
- Try our fine-tuned model at the HuggingFace model hub:
    - **[TIM-BLOOMZ-7b](https://huggingface.co/Lemoooon/TIM-BLOOMZ-7b)**
    - **[TIM-LLaMA-13b](https://huggingface.co/Lemoooon/TIM-LLaMA-13b)**
- Please refer our **[paper](https://arxiv.org/pdf/2307.04408.pdf)** for more detail. 

:star: **Tips** :star:
- When training with Deepspeed ZeRO stage 1/2, we can set --use_low_cpu_mem=True to save memory usage
- After training a model using Deepspeed **ZeRO stage3**, we need to use [sft_reward_training/change_param_name.py](https://github.com/lemon0830/TIM/blob/main/sft_reward_training/change_param_name.py) to perform a transformation of the model's parameter names before inference.

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

- Training data: [train_data/alpaca_reward.json](https://github.com/lemon0830/TIM/blob/main/train_data/alpaca_reward.json), **[train.data.json](https://huggingface.co/datasets/Lemoooon/Train-for-TIM)**

  An essential ingredient of our method is the construction of samples used to provide comparison signals for model learning. In addition to regular translation data, we construct data used for comparison by introducing dictionary information or translation errors

  ![image](https://github.com/lemon0830/TIM/blob/main/images/Fig_data_construct.png)
  
 - test data: [test_data/wmt22](https://github.com/lemon0830/TIM/tree/main/test_data/wmt22), [test_data/flores200](https://github.com/lemon0830/TIM/tree/main/test_data/flores200)

 ### Data Construction for TIM
 We modify add_noisy.py in [noisy-text](https://github.com/valentinmace/noisy-text).
 
 - [add noisy](https://github.com/lemon0830/TIM/tree/main/noisy-text)

 We use the following setting in our paper: 
 ```
    python add_noise.py data/example --delete_probability 0.15 --replace_probability 0.15  --filler_token '' --permutation_range 1
 ```

 Then, you can run [[run_reward.sh]](https://github.com/lemon0830/TIM/blob/main/data_processing/run_reward.sh) to get the final training data for TIM.
 
 ### Instruct Tuning with TIM
 
 We modify `run_clm.py` and `Trainer` in transformers, and `utils` for LoRA in Deepspeed-Chat.
 In addition to vanilla fine-tuning all model parameters, parameter-efficient fine-tuning methods are specially proposed for large language models such as prefix tuning and LoRA. 
 We adopt three different strategies for tuning the models, listed in descending order from the number of fine-tuned parameters.
 
 **(1) LoRA: Tuning with Low-rank Matrices**
 
 - [sft_reward_training/run_lora.sh](https://github.com/lemon0830/TIM/blob/main/sft_reward_training/run_lora.sh)
 
 ```
    --only_optimize_lora    # if True, only optimizing the parameters of LoRA
    --lora_dim 8  
    --lora_alpha 16 
    --lora_droppout 0.05 
    --lora_module_name ${LORA_MODULE_NAME} 
 ```

 **(2) FixEmb: Tuning with Embedding Fixed**
 
 - [sft_reward_training/run_fixemb.sh](https://github.com/lemon0830/TIM/blob/main/sft_reward_training/run_fixemb.sh)
 
 ```
    --only_optimize_layers "9" "8" "7" "6" "5" "4" "3" "2" "1" "0" 
 ```
 
 **(2) Full: Tuning with Full Parameters**
 
 - [sft_reward_training/run_full.sh](https://github.com/lemon0830/TIM/blob/main/sft_reward_training/run_full.sh)

### Deepspeed Config

- deepspeed_config/ds_config.json, deepspeed_config/ds_config_stage2.json, deepspeed_config/ds_config_stage3.json

### Inference 

 - inference/infer_bloom.py, inference/infer_llama.py
 
 - [inference/run_test_bloomz.sh](https://github.com/lemon0830/TIM/blob/main/inference/run_test_bloomz.sh)
 
 ```
    --rootmodel   # if LoRA, the path of the foundation model
    --ifhint      # add note indicates no mistakes in the hypothesize
    --ifsample    # if true, use sample else beam search for inference
    --ifreranking # use the preference score to select a preferred hypothesize in candidates
    --vocab       # the dictionary for dict-guided inference
    --reverse     # whether reverse the src language and tgt language when loading the dictionary
 ```
 
### Experimental Results

We evaluate TIM's performance on the WMT and FLORES-200 dev-test tasks, comprising four language pairs.

<div align="center">
<img src="https://github.com/lemon0830/TIM/blob/main/images/Fig_Results.png" width="70%" alt="result"/>
</div>

### Citation
Please kindly cite our paper if you find it helpful:

```ruby
@inproceedings{zeng2023tim,
  title={TIM: Teaching LM to Translate with Comparison}, 
  author={Jiali Zeng and Fandong Meng and Yongjing Yin and Jia Zhou},
  booktitle = {ArXiv},
  year      = {2023},
  url = {https://arxiv.org/pdf/2307.04408.pdf}
}
```
 
