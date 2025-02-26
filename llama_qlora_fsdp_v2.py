import os
import torch
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
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)

# Using Unsloth's pre-quantized model
MODEL_ID = "unsloth/llama-2-7b-bnb-4bit"  # Replace with actual Unsloth model
HF_TOKEN = "your_huggingface_token"  # Replace with actual token

def setup_fsdp_envs():
    """Set FSDP environment variables for optimal performance"""
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_SHARDING_STRATEGY"] = "1"  # FULL_SHARD
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_POST"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # For 2x T4 GPUs

def get_policies():
    """Configure FSDP policies"""
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
    
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        mixed_precision_policy=mixed_precision_policy,
        cpu_offload=CPUOffload(offload_params=True),
        state_dict_type=StateDictType.FULL_STATE_DICT,
        limit_all_gathers=True,
    )
    
    return fsdp_plugin

def get_model_and_tokenizer():
    """Load pre-quantized model from Unsloth"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map=None,  # Important for FSDP
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def prepare_training_args(output_dir, fsdp_plugin):
    """Configure training arguments with FSDP"""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        num_train_epochs=3,
        max_steps=200,
        fp16=True,
        fsdp=fsdp_plugin,
        fsdp_config={
            "fsdp_sync_module_states": True,
            "fsdp_offload_params": True,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        },
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused"
    )

def main():
    # Setup FSDP environment
    setup_fsdp_envs()
    
    # Initialize accelerator with FSDP plugin
    fsdp_plugin = get_policies()
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    
    # Sample training data - replace with your actual dataset
    train_data = [
        {
            "prompt": "Write a poem about AI",
            "response": "In circuits deep and networks wide,\nSilicon dreams do now reside..."
        }
    ]
    
    # Prepare dataset
    df = pd.DataFrame(train_data)
    df["text"] = df.apply(
        lambda x: f"<|im_start|>user\n{x['prompt']}<|im_end|>\n<|im_start|>assistant\n{x['response']}<|im_end|>\n",
        axis=1
    )
    dataset = Dataset.from_pandas(df)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # Specific to Llama architecture
    )
    
    # Setup training arguments
    training_args = prepare_training_args("llama-qlora-fsdp-output", fsdp_plugin)
    
    # Initialize trainer with FSDP support
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
        formatting_func=lambda x: x["text"]
    )
    
    # Prepare for distributed training
    with accelerator.main_process_first():
        trainer.train()
    
    # Save the final model
    if accelerator.is_main_process:
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
