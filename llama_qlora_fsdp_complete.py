import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaForCausalLM,
    GenerationConfig
)
from transformers.trainer_pt_utils import get_parameter_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from huggingface_hub import notebook_login
import random
import numpy as np
from functools import partial
from time import perf_counter

# Model ID
model_id = "meta-llama/Meta-Llama-3.1-8B"

def get_model_and_tokenizer(model_id):
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
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

def generate_response(model, tokenizer, user_input, formatted_prompt=None):
    if formatted_prompt is None:
        prompt = user_input
    else:
        prompt = formatted_prompt(user_input)
        
    generation_config = GenerationConfig(
        penalty_alpha=0.6,
        do_sample=True,
        top_k=5,
        temperature=0.5,
        repetition_penalty=1.2,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id
    )
    
    start_time = perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_time = perf_counter() - start_time
    
    print(response)
    print(f"Time taken for inference: {round(output_time,2)} seconds")
    return response

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def malloc_in_gb():
    return torch.cuda.memory_allocated() / 1024**3

def free_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_model_size_config(model_size):
    if model_size == "DEBUG":
        model_size_config = dict(hidden_size=128,
                                num_hidden_layers=2,
                                num_attention_heads=2,
                                num_key_value_heads=2,
                                intermediate_size=256)
    elif model_size == "60M":
        model_size_config = dict(hidden_size=512,
                                num_hidden_layers=4,
                                num_attention_heads=4,
                                num_key_value_heads=4,
                                intermediate_size=1024)
    elif model_size == "120M":
        model_size_config = dict(hidden_size=768,
                                num_hidden_layers=12,
                                num_attention_heads=12,
                                num_key_value_heads=12,
                                intermediate_size=1536)
    elif model_size == "290M":
        model_size_config = dict(hidden_size=1024,
                                num_hidden_layers=12,
                                num_attention_heads=16,
                                num_key_value_heads=16,
                                intermediate_size=4096)
    elif model_size == "1B":
        model_size_config = dict(hidden_size=2048,
                                num_hidden_layers=24,
                                num_attention_heads=16,
                                num_key_value_heads=16,
                                intermediate_size=4096)
    elif model_size == "7B":
        model_size_config = {}
    return model_size_config

def create_model(model_size="1B"):
    model_size_config = get_model_size_config(model_size)
    config = LlamaConfig()
    config.update(model_size_config)
    model = LlamaForCausalLM(config)
    return model

def profile_model(create_model_func, inference=False, save_filename="mem_profile.pickle"):
    set_seed(42)
    torch.cuda.memory._record_memory_history()
    
    # Sample inputs for profiling
    inputs = [torch.randint(0, 32000, (1, 512)).cuda()]
    
    for x in inputs:
        print(f"Input Size:{tuple(x.size())}")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        if inference:
            with torch.no_grad():
                model = create_model_func()
                model.to("cuda", torch.bfloat16)
                print(f"Memory allocated [MODEL]: {malloc_in_gb():.3f} GB")
                output = model(x.to("cuda"))
                print(f"Memory allocated [FWD]: {malloc_in_gb():.3f} GB")
        else:
            model = create_model_func()
            model.to("cuda", torch.bfloat16)
            print(f"Memory allocated [MODEL): {malloc_in_gb():.3f} GB")
            output = model(x.to("cuda"))
            print(f"Memory allocated [FWD]: {malloc_in_gb():.3f} GB")            
            output.logits.mean().backward()
            print(f"Memory allocated [BWD]: {malloc_in_gb():.3f} GB")
        end.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(end) / 1000
        print(f"Elapsed time: {secs:.3f}\n\n")
        output, model = None, None
        free_memory()
    torch.cuda.memory._dump_snapshot(save_filename)
    print(f"Memory allocated [finish]: {malloc_in_gb():.3f} GB")

def replace_with_bnb_4bit_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    quant_storage=torch.uint8, 
    keep_trainable=False,
):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                model._modules[name] = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    quantization_config.bnb_4bit_compute_dtype,
                    compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                    quant_type=quantization_config.bnb_4bit_quant_type,
                    quant_storage=quant_storage
                )
                has_been_replaced = True
                model._modules[name].source_cls = type(module)
                if keep_trainable:
                    model._modules[name].requires_grad_(True)
                else:
                    model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_bnb_4bit_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                quant_storage=quant_storage,
                keep_trainable=keep_trainable
            )
        current_key_name.pop(-1)
    return model, has_been_replaced

def setup_distributed():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def get_policies(model):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    return transformer_auto_wrap_policy(transformer_layer_cls={LlamaDecoderLayer})

def setup_fsdp_model(model):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
    
    wrap_policy = get_policies(model)
    
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=True),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn
    )
    
    return model

def main():
    # Login to HuggingFace
    notebook_login()
    
    # Load base model and tokenizer
    print("Loading base model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(model_id)
    
    # Test generation before training
    if dist.get_rank() == 0:
        test_prompt = "What is machine learning?"
        print("\nTesting generation before training:")
        generate_response(model, tokenizer, test_prompt)
    
    # Setup distributed training
    setup_distributed()
    
    # Profile model memory before training
    if dist.get_rank() == 0:
        profile_model(lambda: model, inference=True, save_filename="pre_train_profile.pickle")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Setup FSDP
    model = setup_fsdp_model(model)
    
    # Load dataset (example using alpaca)
    dataset = load_dataset("tatsu-lab/alpaca")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llama3-qlora-fsdp",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        ddp_backend="nccl",
        gradient_checkpointing=True,
        report_to="wandb"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Test generation after training
    if dist.get_rank() == 0:
        test_prompt = "What is machine learning?"
        print("\nTesting generation after training:")
        generate_response(model, tokenizer, test_prompt)
    
    # Profile model memory after training
    if dist.get_rank() == 0:
        profile_model(lambda: model, inference=True, save_filename="post_train_profile.pickle")
    
    # Save model
    if trainer.is_world_process_zero():
        trainer.save_model()

if __name__ == "__main__":
    main()
