#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import sys
from typing import Optional, Dict
from functools import partial
import datasets
import torch
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM, 
    Trainer, 
    TrainingArguments,
    EvalPrediction
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model

from train.arguments import ModelArguments, DataArguments
from train.data import CustomDataset
from train.eval_callback import CodeGenerationEvalCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for evaluation.
    Currently computes perplexity and training accuracy (token-level).
    """
    predictions, labels = eval_pred
    
    # predictions are logits, shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # Both are numpy arrays from the trainer
    
    # Shift so that tokens < n predict n
    shift_logits = predictions[..., :-1, :]
    shift_labels = labels[..., 1:]
    
    # Flatten the tokens
    shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.reshape(-1)
    
    # Compute metrics
    mask = (shift_labels != -100)
    if mask.sum() > 0:
        # Get predicted tokens
        pred_tokens = np.argmax(shift_logits, axis=-1)
        # Compute accuracy only on non-ignored tokens
        correct = (pred_tokens == shift_labels) & mask
        accuracy = correct.sum() / mask.sum()
        
        # Compute loss (cross-entropy) using log-softmax for numerical stability
        # Use log-sum-exp trick for numerical stability
        max_logits = np.max(shift_logits, axis=-1, keepdims=True)
        log_probs = shift_logits - max_logits - np.log(np.sum(np.exp(shift_logits - max_logits), axis=-1, keepdims=True) + 1e-10)
        
        # Select log-probabilities for correct tokens
        selected_log_probs = log_probs[np.arange(len(shift_labels)), shift_labels]
        loss = -selected_log_probs[mask].mean()
        perplexity = np.exp(loss)
    else:
        accuracy = 0.0
        loss = 0.0
        perplexity = 1.0
    
    return {
        "eval_perplexity": float(perplexity),
        "eval_loss": float(loss),
        "eval_accuracy": float(accuracy),
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Note: WandB will be automatically initialized by Trainer when report_to="wandb"
    # The Trainer will use environment variables (WANDB_PROJECT, WANDB_RUN_NAME, etc.)
    # or TrainingArguments (run_name, etc.) for configuration
    if training_args.report_to and 'wandb' in training_args.report_to:
        wandb_project = os.environ.get('WANDB_PROJECT', 'orlm_sft_experiments')
        wandb_run_name = os.environ.get('WANDB_RUN_NAME', training_args.run_name)
        logger.info(f"WandB will be initialized by Trainer with project: {wandb_project}, run: {wandb_run_name}")

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None, 
        "trust_remote_code": True, 
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this finetuning script."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True, 
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this finetuning script."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True, 
        )
    else:
        logger.warning("No pretrained model_name_or_path is given. Training new model from scratch.")
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            logging.info("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_lora:
        logger.info("Initializing LORA model...")
        logger.info(f"LoRA Rank: {model_args.lora_rank}, Alpha: {model_args.lora_alpha}, Dropout: {model_args.lora_dropout}")
        logger.info(f"LoRA Target Modules: {model_args.lora_target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=model_args.lora_rank, 
            lora_alpha=model_args.lora_alpha, 
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        logger.info(f"LoraConfig: {peft_config}")
        logger.info(f"Applying LoRA with rank={model_args.lora_rank} to model...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # Ensure model is in training mode for LoRA
        model.train()
        # Verify trainable parameters have requires_grad=True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
        logger.info(f"LoRA rank {model_args.lora_rank} successfully applied - trainable params: {trainable_params:,}")
        
        # Verify LoRA adapters are actually in the model
        lora_modules = [name for name, module in model.named_modules() if 'lora' in name.lower()]
        logger.info(f"Found {len(lora_modules)} LoRA modules in model")
        if len(lora_modules) > 0:
            logger.info(f"Sample LoRA modules: {lora_modules[:5]}")  # Show first 5
        
        # Verify that base model parameters are frozen
        base_trainable = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'lora' not in name.lower())
        lora_trainable = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'lora' in name.lower())
        logger.info(f"Base model trainable params: {base_trainable:,}, LoRA trainable params: {lora_trainable:,}")
        if base_trainable > 0:
            logger.warning(f"⚠️  WARNING: {base_trainable:,} base model parameters are trainable! This should be 0 for LoRA.")
        if lora_trainable == 0:
            logger.error(f"❌ ERROR: No LoRA parameters are trainable! LoRA adapters may not be properly initialized.")

    # set up datasets
    train_dataset = CustomDataset(training_args, data_args, model_args, tokenizer, is_eval=False)
    
    # Set up evaluation dataset if provided
    eval_dataset = None
    if data_args.eval_dataset_name_or_path:
        # Create a separate data_args for evaluation
        eval_data_args = DataArguments(
            train_dataset_name_or_path=data_args.eval_dataset_name_or_path,
            max_seq_length=data_args.max_seq_length,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            overwrite_cache=data_args.overwrite_cache,
            max_eval_samples=data_args.max_eval_samples,
        )
        eval_dataset = CustomDataset(training_args, eval_data_args, model_args, tokenizer, is_eval=True)
        logger.info(f"Evaluation dataset loaded: {len(eval_dataset)} examples")

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)

    # Set up code generation evaluation callback if eval datasets are provided
    callbacks = []
    if hasattr(data_args, 'eval_datasets') and data_args.eval_datasets:
        # Parse eval_datasets string (format: "dataset1:path1,dataset2:path2")
        eval_datasets_dict = {}
        for item in data_args.eval_datasets.split(','):
            if ':' in item:
                name, path = item.split(':', 1)
                eval_datasets_dict[name.strip()] = path.strip()
        
        if eval_datasets_dict:
            code_eval_callback = CodeGenerationEvalCallback(
                eval_datasets=eval_datasets_dict,
                tokenizer=tokenizer,
                model_name_or_path=model_args.model_name_or_path,  # For vLLM initialization
                max_new_tokens=10000,  # Match eval.py default
                temperature=0.0,
                numerical_tolerance=0.001,
                eval_steps=training_args.eval_steps if hasattr(training_args, 'eval_steps') and training_args.eval_steps else 50,
                max_eval_samples=10,  # Only evaluate on 10 examples per dataset (30 total instead of 90)
                skip_code_execution=False,  # Set to True to skip code execution for even faster evaluation
                timeout=30,  # Match run_eval.sh timeout
                tensor_parallel_size=1,  # vLLM tensor parallelism (use 1 for training-time eval)
                gpu_memory_utilization=0.8,  # vLLM GPU memory utilization
                use_vllm=True,  # Use vLLM for fast inference (same as eval.py)
            )
            callbacks.append(code_eval_callback)
            logger.info(f"Code generation evaluation callback added for datasets: {list(eval_datasets_dict.keys())}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        preprocess_logits_for_metrics=None,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=callbacks,
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Always save final model at end of training
    # This ensures we can evaluate the trained model later
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()
    logger.info(f"Final model and state saved to {training_args.output_dir}")

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)  # Always save metrics for analysis


if __name__ == "__main__":
    main()
