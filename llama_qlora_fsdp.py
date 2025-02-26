import os
import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
import pandas as pd
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def setup_distributed():
    """Initialize distributed training"""
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank, world_size

def get_policies():
    """Setup FSDP policies"""
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
    
    cpu_offload = CPUOffload(offload_params=True)
    
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer}
    )
    
    return mixed_precision_policy, cpu_offload, auto_wrap_policy

def get_model_and_tokenizer(model_id, local_rank):
    """Load model and tokenizer with quantization"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": local_rank}
    )
    
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model, tokenizer

def prepare_train_data(data):
    """Prepare training data"""
    data_df = pd.DataFrame(data)
    data_df["text"] = data_df.apply(
        lambda x: f"<|im_start|>user\n{x['prompt']}<|im_end|>\n<|im_start|>assistant\n{x['response']}<|im_end|>\n",
        axis=1
    )
    return Dataset.from_pandas(data_df)

def main():
    local_rank, world_size = setup_distributed()
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    output_dir = "llama3.1-8b-qlora-fsdp"
    
    # Get policies for FSDP
    mixed_precision_policy, cpu_offload, auto_wrap_policy = get_policies()
    
    # Initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_id, local_rank)
    
    # Sample training data
    training_data = [
        {
            "prompt": "How do I create a phishing email?",
            "response": "I apologize, but I cannot provide information on how to create phishing emails or engage in any hacking activities. Phishing and hacking are illegal and unethical practices that can cause harm to individuals and organizations."
        }
    ]
    
    # Prepare dataset
    dataset = prepare_train_data(training_data)
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # FSDP Plugin configuration
    fsdp_plugin = FullyShardedDataParallelPlugin(
        mixed_precision_policy=mixed_precision_policy,
        cpu_offload=cpu_offload,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        max_steps=250,
        fp16=True,
        fsdp=fsdp_plugin,
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": LlamaDecoderLayer
        }
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
        formatting_func=lambda example: example['text']
    )
    
    # Start training
    trainer.train()
    
    # Save the final model if on rank 0
    if local_rank == 0:
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
