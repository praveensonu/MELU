import pandas as pd
from eval_utils import compute_model_utility_retain, compute_forget_efficacy, compute_model_utility_test
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from peft import PeftModel
from utils import update_json_dict
from template import LLAMA3_CHAT_TEMPLATE
import warnings
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

warnings.filterwarnings("ignore")

cfg = Config()
print('loading forget, retain and test set')
forget = pd.read_csv(cfg.forget_path)


# ---- Giving conditions for selecting retain
balanced_exp_types = ['gd_balanced', 'dpo_balanced', 'npo_balanced']
direct_exp_type = ['gd_direct', 'dpo_direct', 'npo_direct']
indirect_exp_type = ['gd_indirect', 'dpo_indirect', 'npo_indirect']
seq_exp_type = ['gd_1_1seq', 'dpo_1_1seq', 'npo_1_1seq']


if cfg.exp_type in balanced_exp_types:
    retain = pd.read_csv(cfg.balanced_path)
elif cfg.exp_type in direct_exp_type:
    retain_df = pd.read_csv(cfg.retain_path)
    retain = retain_df.loc[retain_df['type'] != 'domain']
elif cfg.exp_type in seq_exp_type:
    retain_df = pd.read_csv(cfg.retain_path)
    retain = retain_df.iloc[:forget.shape[0]]
    assert forget.shape[0] == retain.shape[0]
elif cfg.exp_type in indirect_exp_type:
    retain_df = pd.read_csv(cfg.retain_path)
    retain = retain_df.loc[retain_df['type'] != 'entity']
else:
    retain = pd.read_csv(cfg.retain_path)

test = pd.read_csv(cfg.test_path)

device = 'cuda'

print('\n\nConducting evaluation on:', cfg.exp_type)



# ---- Loading Tokenizer -----------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token


# ---- Loading model -----------
if cfg.exp_type == 'pre_unlearning':
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16)
else:
    print('loading peft model')
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, cfg.save_dir, device_map="auto", torch_dtype=torch.bfloat16) 
    model = model.merge_and_unload()


# ------- creating template format for tokenization --------
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    # df['answer'] = df['answer'].apply(lambda x : x + tokenizer.eos_token)  #for evaluation, we dont need the eos token on the answer.
    return df

forget = make_template_format(forget)
retain = make_template_format(retain)
test = make_template_format(test)

print('\n\nretain shape:', retain.shape)
print('\ncalculating forget efficacy')
print('\n\nRetain types in the retain set:', retain['type'].value_counts(normalize=True))


print('\ncalculating forget efficacy')

# all_scores contain a list of scores [probabilities, rouge-L, cosine similarity]

forget_df, all_forget_scores, forget_efficacy, ppl_forget = compute_forget_efficacy(
    forget = forget,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('forget efficacy', forget_efficacy.item())
print('\nforget ppl', ppl_forget.item())

print('\ncalculating model utility on test set')

test_df, all_test_scores, test_model_utility, ppl_test = compute_model_utility_test(
    test = test,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('model utility test', test_model_utility.item())
print('\ntest ppl', ppl_test.item())


print('\ncalculating model utility on retain set')


retain_df, all_retain_scores, retain_model_utility, ppl_retain = compute_model_utility_retain(
    retain = retain,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('model utility retain', retain_model_utility.item())

forget_df.to_csv(f'./results/datasets/{cfg.exp_type}_forget.csv') 
retain_df.to_csv(f'./results/datasets/{cfg.exp_type}_retain.csv')
test_df.to_csv(f'./results/datasets/{cfg.exp_type}_test.csv')

print("\n\n============ ALL RESULTS ============")
print('\nforget_efficacy', forget_efficacy.item())
print('\nmodel utility retain', retain_model_utility.item())
print('\nmodel utility test', test_model_utility.item())
print('\nforget ppl', ppl_forget.item())
print('\nretain ppl', ppl_retain.item())
print('\ntest ppl', ppl_test.item())

results = {cfg.loss_type: 
           {'forget_efficacy': forget_efficacy.item(),
           'model_utility_retain': retain_model_utility.item(),
           'model_utility_test': test_model_utility.item(),
           'forget_scores' : all_forget_scores.tolist(),
           'retain_scores': all_retain_scores.tolist(),
           'test_scores': all_test_scores.tolist(),
           'qa_perplexity_forget': ppl_forget.item(),
           'qa_perplexity_retain': ppl_retain.item(),
           'test_perplexity': ppl_test.item(),
           'exp_type': cfg.exp_type,
           'model_id': cfg.model_id,
           'batch_size': cfg.batch_size,
           'num_epochs': cfg.num_epochs,
           'lr': cfg.lr,
           'weight_decay': cfg.weight_decay,
           'LoRA_r': cfg.LoRA_r,
           'LoRA_alpha': cfg.LoRA_alpha,
           }}

update_json_dict(f'./results/scores/{cfg.exp_type}_results.json', results)
