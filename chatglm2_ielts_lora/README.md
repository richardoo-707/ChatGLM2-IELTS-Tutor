---
library_name: peft
license: other
base_model: /root/autodl-tmp/chatglm2-6b
tags:
- base_model:adapter:/root/autodl-tmp/chatglm2-6b
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: chatglm2_ielts_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# chatglm2_ielts_lora

This model is a fine-tuned version of [/root/autodl-tmp/chatglm2-6b](https://huggingface.co//root/autodl-tmp/chatglm2-6b) on the ielts_finetune dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.18.0
- Transformers 4.57.1
- Pytorch 2.9.1+cu128
- Datasets 4.4.2
- Tokenizers 0.22.1