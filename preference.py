# 1. export CUDA_VISIBLE_DEVICES=3,5
# 2. accelerate launch --multi_gpu --num_processes 2 preference.py

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from pref_utils import *
from pref_data_module import *
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, default_data_collator
from accelerate import  Accelerator
from config import Config
import torch
from peft import  LoraConfig, get_peft_model
from utils import find_all_linear_names
import pandas as pd
import logging
from template import LLAMA3_CHAT_TEMPLATE


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg = Config()

accelerator = Accelerator()

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token


# --- Load policy model ---
# we load it on cpu, let accelerate move it to GPU with accelerate.prepare_model
policy_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=cfg.access_token
    )
print("Base model loaded.")

# --- Apply LoRA on policy model ---
print("Applying LoRA...")
lora_config = LoraConfig(
    r=cfg.LoRA_r,
    lora_alpha=cfg.LoRA_alpha,
    lora_dropout=cfg.LoRA_dropout,
    target_modules=find_all_linear_names(policy_model),
    bias='none',
    task_type='CAUSAL_LM',
)

# Get PEFT model
model = get_peft_model(policy_model, lora_config)
print("PEFT model created.")
model.print_trainable_parameters()
model.config.use_cache = False 

# --- Load reference model ---
ref_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=cfg.access_token
)


forget = pd.read_csv(cfg.forget_path)

if cfg.exp_type == 'dpo_balanced' or cfg.exp_type == 'npo_balanced':
    retain = pd.read_csv(cfg.balanced_path)
else:     
    retain = pd.read_csv(cfg.retain_path)

retain['idk'] = 'idk'

print('\n\nretain shape:', retain.shape)

#creates llama template for the forget and retain dataset 
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    df['answer'] = df['answer'].apply(lambda x : x + tokenizer.eos_token)
    df['idk'] = df['idk'].apply(lambda x : x + tokenizer.eos_token)
    return df
forget = make_template_format(forget)
retain = make_template_format(retain)
print('\n\nforget question - \n',forget['question'][0])

print('\n\nforget answer - \n',forget['answer'][0])

print('\n\nforget IDK - \n', forget['idk'][0])



# ---- Training args ----
training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        overwrite_output_dir= True,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size,
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        eval_strategy= 'no',
        label_names = ['labels'],
        bf16 = True,
        gradient_accumulation_steps= cfg.gradient_accumulation_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
)


# ---- DPO and NPO without retain -------
if cfg.exp_type == 'dpo':
        train_dataset = VanillaDPODataset(forget_data=forget,
                                  tokenizer=tokenizer,
                                  max_length = 256,
                                  question_key='question',
                                  answer_key='answer',
                                  idk_key='idk'
                                  )
        print("\n\n=======Conducting Vanilla DPO Unlearning now=======")
        trainer = VanillaDPOTrainer(
            model=model,       # The model to be trained (policy)
            ref_model=ref_model,      # The frozen reference model
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator
        )

if cfg.exp_type == 'npo':

     train_dataset = VanillaDPODataset(forget_data=forget,
                                  tokenizer=tokenizer,
                                  max_length = 256,
                                  question_key='question',
                                  answer_key='answer',
                                  idk_key='idk'
                                  )

     print("\n\n=======Conducting Vanilla NPO Unlearning now=======")
     trainer = VanillaNPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator
        )

# ---- 1_1seq on DPO and NPO ----
if cfg.exp_type == 'dpo_1_1seq':
    retain_df = retain.iloc[:forget.shape[0]]
    print('\n\nForget shape is:',forget.shape)
    print('\n\nRetain shape is:',retain_df.shape)
    assert forget.shape[0] == retain_df.shape[0]
    train_dataset = ForgetIdkRetainDataset(forget_data=forget,
                                            retain_data=retain_df,
                                            tokenizer = tokenizer,
                                            max_length = 256,
                                            question_key='question',
                                            answer_key='answer',
                                            idk_key='idk'
                                            )
    print('\n\nlength of the dataset', len(train_dataset))
    print("\n\n=======Conducting DPO 1_1 sequential Unlearning now=======")
    trainer = RetainDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
        )


if cfg.exp_type == 'npo_1_1seq':
    retain_df = retain.iloc[:forget.shape[0]]
    print('\n\nForget shape is:',forget.shape)
    print('\n\nRetain shape is:',retain_df.shape)
    assert forget.shape[0] == retain_df.shape[0]
    train_dataset = ForgetIdkRetainDataset(forget_data=forget,
                                        retain_data=retain_df,
                                        tokenizer = tokenizer,
                                        max_length = 256,
                                        question_key='question',
                                        answer_key='answer',
                                        idk_key='idk'
                                        )
    print('\n\nlength of the dataset', len(train_dataset))
    print("\n\n=======Conducting NPO 1_1 sequential Unlearning now=======")
    trainer = RetainNPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        data_collator = default_data_collator,
        gamma = 1.0,
        alpha = 1.0,
    )


# === 1_1random based implementation (TOFU implementation) ===
if cfg.exp_type == 'npo_1_1random':
    print('\n\ncreating the dataset for TOFU type of implementation (dpo_1_1random)')
    train_dataset = ForgetIdkRetainDatasetRandom(forget_data = forget,
                                                  retain_data = retain,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
                                                  )
    print('\n\nlength of the dataset', len(train_dataset))
    trainer = RetainDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
    )


if cfg.exp_type == 'npo_1_1random':
    print('\n\ncreating the dataset for TOFU type of implementation (npo_1_1random)')
    train_dataset = ForgetIdkRetainDatasetRandom(forget_data = forget,
                                                  retain_data = retain,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
                                                  )
    print('\n\nlength of the dataset', len(train_dataset))
    trainer = RetainNPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
    )

if cfg.exp_type == 'dpo_direct':
    print('\n\ncreating the dataset for dpo_direct')
    retain_df = retain.loc[retain['type'] != 'domain']
    print('\n\nRemoved direct, retain shape is:',retain_df.shape)
    print('\n\nDomain Exclusive type:', retain_df['type'].value_counts(normalize=True))
    train_dataset = CyclicForgetIdkRetainDataset(forget_data = forget,
                                                  retain_data = retain_df,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
     )    
    print('\n\nlength of the dataset', len(train_dataset))
    print("\n\n=======Conducting DPO Direct Unlearning now=======")
    trainer = RetainDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
    )



if cfg.exp_type == 'npo_direct':
    print('\n\ncreating the dataset for npo_direct')
    retain_df = retain.loc[retain['type'] != 'domain']
    print('\n\nRemoved Domain, retain shape is:',retain_df.shape)
    print('\n\Indirect Exclusive:', retain_df['type'].value_counts(normalize=True))
    train_dataset = CyclicForgetIdkRetainDataset(forget_data = forget,
                                                  retain_data = retain_df,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
     )    
    print('\n\nlength of the dataset', len(train_dataset))
    print("\n\n=======Conducting NPO Direct Unlearning now=======")
    trainer = RetainNPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
    )


if cfg.exp_type == 'dpo_indirect':
    print('\n\ncreating the dataset for dpo_indirect')
    retain_df = retain.loc[retain['type'] != 'entity']
    print('\n\nRemoved direct connections, retain shape is:',retain_df.shape)
    print('\n\Direct Exclusive data:', retain_df['type'].value_counts(normalize=True))
    train_dataset = CyclicForgetIdkRetainDataset(forget_data = forget,
                                                  retain_data = retain_df,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
     )    
    print('\n\nlength of the dataset', len(train_dataset))
    print("\n\n=======Conducting DPO Indirect Unlearning now=======")
    trainer = RetainDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
    )


if cfg.exp_type == 'npo_indirect':
    print('\n\ncreating the dataset for npo_indirect')
    retain_df = retain.loc[retain['type'] != 'entity']
    print('\n\nRemoved direct connections, retain shape is:',retain_df.shape)
    print('\n\direct Exclusive data:', retain_df['type'].value_counts(normalize=True))
    train_dataset = CyclicForgetIdkRetainDataset(forget_data = forget,
                                                  retain_data = retain_df,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
     )    
    print('\n\nlength of the dataset', len(train_dataset))
    print("\n\n=======Conducting NPO Indirect Unlearning now=======")
    trainer = RetainNPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
    )

if cfg.exp_type == 'dpo_cyclic' or cfg.exp_type == 'dpo_balanced':
     train_dataset = CyclicForgetIdkRetainDataset(forget_data = forget,
                                                  retain_data = retain,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
     )
     print('\n\nlength of the dataset', len(train_dataset))
     print(f"\n\n=======Conducting {cfg.exp_type} Unlearning now=======")
     trainer = RetainDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
     )

if cfg.exp_type == 'npo_cyclic' or cfg.exp_type == 'npo_balanced':
    train_dataset = CyclicForgetIdkRetainDataset(forget_data = forget,
                                                  retain_data = retain,
                                                  tokenizer = tokenizer,
                                                  max_length = 256,
                                                  question_key='question',
                                                  answer_key='answer',
                                                  idk_key='idk'
     )
    print('\n\nlength of the dataset', len(train_dataset))
    print(f"\n\n=======Conducting {cfg.exp_type} Unlearning now=======")
    trainer = RetainNPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
     )


# === MELU unlearning =====
if cfg.exp_type == 'dpo_melu':
    melu_data = pd.read_csv(cfg.melu_path)
    def make_template_format(df):
        df['question_forget'] = df['question_forget'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
        df['answer_forget'] = df['answer_forget'].apply(lambda x : x + tokenizer.eos_token)
        df['question_retain'] = df['question_retain'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
        df['answer_retain'] = df['answer_retain'].apply(lambda x : x + tokenizer.eos_token)
        return df

    melu_data = make_template_format(melu_data)
    print('\n\Melu dataframe shape is:',melu_data.shape)
    print('\n\nForget question and answer\n',melu_data['question_forget'][0], melu_data['answer_forget'][0])
    print('\n\nRetain df question and answer\n',melu_data['question_retain'][0], melu_data['answer_retain'][0])
    train_dataset = MELUForgetIdkRetainDataset(
                                            data = melu_data,
                                            tokenizer = tokenizer,
                                            max_length = 256,
                                            )
    print("\n\n=======Conducting MELU DPO Unlearning now=======")
    trainer = RetainDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
        )


if cfg.exp_type == 'npo_melu':
    melu_data = pd.read_csv(cfg.melu_path)
    def make_template_format(df):
        df['question_forget'] = df['question_forget'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
        df['answer_forget'] = df['answer_forget'].apply(lambda x : x + tokenizer.eos_token)
        df['question_retain'] = df['question_retain'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
        df['answer_retain'] = df['answer_retain'].apply(lambda x : x + tokenizer.eos_token)
        return df

    melu_data = make_template_format(melu_data)
    print('\n\MELU dataframe shape is:',melu_data.shape)
    print('\n\nForget question and answer\n',melu_data['question_forget'][0], melu_data['answer_forget'][0])
    print('\n\nRetain df question and answer\n',melu_data['question_retain'][0], melu_data['answer_retain'][0])
    train_dataset = MELUForgetIdkRetainDataset(
                                            data = melu_data,
                                            tokenizer = tokenizer,
                                            max_length = 256,
                                            )

    print("\n\n=======Conducting NPO MELU Unlearning now=======")
    trainer = RetainNPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            beta=0.1,
            data_collator = default_data_collator,
            gamma = 1.0,
            alpha = 1.0,
     )


trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
if training_args.local_rank <= 0:
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"Rank {training_args.local_rank}: Tokenizer saved.")
else:
    tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')