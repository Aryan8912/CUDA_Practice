# QLoRA with FSDP2 Training

This repository demonstrates how to finetune Llama 3.1 8B using QLoRA (Quantized Low-Rank Adaptation) with FSDP2 (Fully Sharded Data Parallel v2) on multiple GPUs.

## Features

- 4-bit quantization using NF4 format
- FSDP2 for distributed training
- Flash Attention 2 support
- CPU offloading
- Activation checkpointing
- Mixed precision training

## Running on Kaggle

1. Create a new notebook with T4 GPU x2 accelerator
2. Install dependencies:
```bash
!pip install -r requirements.txt
```

3. Run the training script:
```bash
!torchrun --nproc_per_node=2 llama_qlora_fsdp.py \
    --model_name_or_path "unsloth/llama-2-7b-bnb-4bit" \
    --output_dir "./output" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --bf16 True \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 3
```

## Implementation Details

- Uses FSDP2's latest features including CPU offloading and activation checkpointing
- Implements efficient memory management through proper sharding strategies
- Utilizes Flash Attention 2 for faster attention computation
- Employs mixed precision training with bf16
- Implements proper gradient checkpointing for memory efficiency

## Memory Usage

With 2x Tesla T4 GPUs (16GB each):
- Model is quantized to 4-bit (NF4)
- LoRA adapters are in bf16
- Activation checkpointing reduces memory usage
- CPU offloading helps manage memory peaks

## Notes

- Make sure to have enough CPU RAM (>32GB recommended) when using CPU offloading
- The implementation is tested and optimized for Tesla T4 GPUs
- Training metrics are logged to wandb for monitoring
