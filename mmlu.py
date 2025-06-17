import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
from config import Config
from tqdm.auto import tqdm
tqdm.pandas()


cfg = Config()


# ---- Loading Tokenizer -----------
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---- Loading Model -----------
print('loading peft model')
base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, cfg.save_dir, device_map="auto", torch_dtype=torch.bfloat16) 
model = model.merge_and_unload()

# ---- Loading MMLU Data -----------
splits = {'test': 'all/test-00000-of-00001.parquet', 'validation': 'all/validation-00000-of-00001.parquet', 'dev': 'all/dev-00000-of-00001.parquet', 'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])


def make_prompt(question, choices):
    choices = list(choices)
    return f"""You are a multiple-choice QA assistant.
Given a question and exactly four answer choices labeled A, B, C, and D, reply with **only** the single letter of the correct answer (A, B, C, or D), and nothing else.

Question: {question}
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
Answer:"""


def generate_answer(prompt, tokenizer, model):
    with torch.no_grad():
        ips = tokenizer(prompt, return_tensors='pt', padding=True).to('cuda')
        out = model.generate(
            **ips,
            max_new_tokens=10,     
            do_sample=False,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id
        )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def generate_answer_letter(prompt, tokenizer, model):
    raw = generate_answer(prompt, tokenizer, model)      
    m = re.search(r'([A-D])\s*$', raw.strip())
    return m.group(1) if m else raw.strip()


def answer_for_row(row):
    prompt = make_prompt(row['question'], row['choices'])
    return generate_answer_letter(prompt, tokenizer, model)

df['temp_answers'] = df.progress_apply(answer_for_row, axis=1)
num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
df['answer_letter'] = df['answer'].map(num_to_letter)

df[cfg.loss_type] = df['temp_answers'].str.extract(r'Answer:\s*([A-D])', expand=False).str.strip()
df[cfg.loss_type] = df[cfg.loss_type].fillna(df['temp_answers'])

df['correct'] = df[cfg.loss_type] == df['answer_letter']
accuracy = df['correct'].mean()
print(f'Accuracy: {accuracy:.2%}')

df.to_csv(f'./results/scores/{cfg.loss_type}_mmlu.csv', index = False)
