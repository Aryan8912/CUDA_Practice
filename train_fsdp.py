# Install required packages
!pip install accelerate peft bitsandbytes transformers trl flash-attn

# Import necessary libraries
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import write_basic_config
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallelPlugin,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp import MixedPrecision
import torch.distributed as dist
from flash_attn.flash_attn_interface import flash_attn_func
import os

def verify_kaggle_environment():
    """Verify we're running on Kaggle with 2x T4 GPUs"""
    assert torch.cuda.device_count() == 2, "This script requires exactly 2 GPUs"
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        assert "T4" in gpu_name, f"GPU {i} is {gpu_name}, but Tesla T4 is required"
    print("✓ Verified Kaggle T4 environment")

# Define model ID
model_id = "meta-llama/Meta-Llama-3.1-8B"

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8  # More memory efficient
    )

    # Use init_empty_weights for more efficient model loading
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="balanced",  # Enable pipeline parallelism
            use_flash_attention_2=True,  # Enable Flash Attention 2
            torch_dtype=torch.float16
        )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer

def verify_loss_equivalence(trainer, accelerator):
    """Verify loss equivalence across GPUs"""
    if accelerator.is_main_process:
        print("Verifying loss equivalence across GPUs...")
    
    # Run a single training step
    outputs = trainer.training_step(trainer.model, next(trainer.get_train_dataloader()))
    loss = outputs.loss
    
    # Gather losses from all GPUs
    gathered_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_losses, loss)
    
    if accelerator.is_main_process:
        max_diff = max([torch.abs(gathered_losses[0] - l).item() for l in gathered_losses[1:]])
        print(f"Maximum loss difference across GPUs: {max_diff}")
        assert max_diff < 1e-5, "Loss is not equivalent across GPUs!"

def main():
    # Verify Kaggle environment
    verify_kaggle_environment()

    # Initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_id)

    # Prepare training data
    training_data = [{"prompt": "How do I create a phishing email?", "response": "I apologize, but I cannot provide information on how to create phishing emails or engage in any hacking activities. Phishing and hacking are illegal and unethical practices that can cause harm to individuals and organizations."}]
    dataset = Dataset.from_dict(training_data)
    
    # Define LoRA configuration with target modules
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Specific target modules
    )

    # Define training arguments optimized for T4 GPUs
    training_arguments = TrainingArguments(
        output_dir="llama3.1-8b-finetuned",
        per_device_train_batch_size=2,  # Reduced for T4 GPUs
        gradient_accumulation_steps=8,   # Increased for stability
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        max_steps=250,
        fp16=True,
        push_to_hub=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": True,
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_cpu_offload": True,
        },
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        report_to="wandb",  # For loss monitoring
    )

    # Initialize Accelerator with enhanced FSDP plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
        mixed_precision="fp16",
        fsdp_plugin=fsdp_plugin
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Verify loss equivalence
    verify_loss_equivalence(trainer, accelerator)

    # Train the model
    trainer.train()

    # Save the model
    if accelerator.is_main_process:
        trainer.save_model("llama3.1-8b-finetuned")

if __name__ == "__main__":
    main()
