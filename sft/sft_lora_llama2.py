import os
import torch

import datasets
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, # quantization
    TrainingArguments,
    pipeline,
    logging
)   
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Base llama 2 model
BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf" 

# Instruction dataset
DATASET = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model
new_model = "model_checkpoint/llama-2-7b-chat-latest"

def run_infer(inst: str):
    # fetch the latest checkpoint for inference

    adapter_model = 'lora_checkpoint/llama-2-7b-chat-latest'

    if os.path.exists(adapter_model):
        pass
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL,
                        low_cpu_mem_usage=True,
                        return_dict=True,
                        torch_dtype=torch.float16,
                        device_map={"": 0},
                    )
    
    peft_model = PeftModel.from_pretrained(base_model, adapter_model)
    peft_model = peft_model.merge_and_unload()

    # instantiate the tokenizer
    tokenizer = AutoTokenizer(adapter_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = transformers.pipeline(
        "text-generation",
        model=peft_model,
        device_map={"":0}
    )
    
    seqs = pipe(
        inst,
        do_sample = True,
        top_k = 10,
        num_return_sequences = 1,
        eos_token_id = tokenizer.eos_token_id,
        max_length = 200
    )

    seqs_text = []
    for seq in seqs:
        seqs_text.append(seq[0]['generated_text'])
    return seqs_text


def run_sft(with_data: bool = False):

    # downloading the dataset from huggingface, for testing only
    if with_data is False:
        dataset = load_dataset(DATASET, split="train")
        temp_dict = dataset[:10]
        dataset = datasets.Dataset.from_dict(temp_dict)
    else:
        dataset = load_dataset('csv', ['fine_tune_data.csv'])
    # instantiate the model
    model = AutoModelForCausalLM(BASE_MODEL, trust_remote_code=True)

    # instantiate the tokenizer
    tokenizer = AutoTokenizer(BASE_MODEL)

    # quantization configuration 
    compute_type = getattr(torch, 'float16')
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = compute_type,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False
    )

    # model configuration
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map={"":0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # lora configuration
    peft_params = LoraConfig(
        lora_alpha = 20,
        r=64,
        lora_dropout = 0.1,
        bias = 'none',
        task_type = 'CAUSAL_LM' # what the different the adapters of different task_type
    )

    # training configuration
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4, # batch per gpu for training
        gradient_accumulation_steps=1,
        gradient_checkpointing=True, # enable gradient checkpoint to save vram
        optim="paged_adamw_32bit",
        save_steps=25, # save checkpoint every number of steps
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant", # learning rate schedule 
        report_to="tensorboard",
    )

    # using supervised fine-tuning trainer from hugging for easy fine-tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    try:
        trainer.train()
    except Exception as ex:
        print("Error during fine-tuning")
        logging.info(ex)

    print("Done fine-tuning")

    print("Saving model")
    # This only save the lora checkpoint
    trainer.tokenizer.save_pretrained(new_model)
    trainer.model.save_pretrained(new_model)
    # clear gpu memory
    del model
    del trainer
    import gc
    gc.collect()
    print("Done saving model")

# Preset configuration for the modules used in this 
if __name__ == "__main__":
    run_sft()
