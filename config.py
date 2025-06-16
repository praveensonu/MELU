
class Config_ft:
    def __init__(self):
        super(Config_ft, self).__init__()
        self.model_id       = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.access_token   = ''
        self.LoRA_r         = 64
        self.LoRA_alpha     = 128
        self.LoRA_dropout   = 0.05
        self.lr             = 2e-05
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj']
        self.batch_size     = 8
        self.gradient_accumulation_steps = 1
        self.num_epochs     = 10
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.exp_type       = 'finetuned'
        self.model_name    = 'llama_3_1_8b'
        self.save_dir       = 'outputs/llama_3_1_8b_finetuned'
        self.max_length     = 256

class Config:
    def __init__(self):
        super(Config, self).__init__()
        # Methods gradient based [grad_ascent,gd_1_1seq, gd_1_1random, gd_direct, gd_indirect, gd_balanced, gd_cyclic, gd_melu]
        # dpo based [dpo, dpo_1_1_seq, dpo_1_1random, dpo_direct, dpo_indirect, dpo_balanced, dpo_cyclic, dpo_melu]
        # npo based [npo, npo_1_1_seq, npo_1_1random, npo_direct, npo_indirect, npo_balanced, npo_cyclic, npo_melu]
        self.loss_type      = 'gd_1_1random' # (1_1seq, 1_1random, direct, indirect, balanced, cyclic, melu)
        self.access_token   = '' # please add your huggingface access token
        self.model_id       = 'outputs/llama_3_1_8b_finetuned' # finetuned model path
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 2e-05
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
        self.batch_size     = 1
        self.gradient_accumulation_steps = 4
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 256
        self.exp_type       = self.loss_type # (1_1seq, 1_1random, direct, indirect, balanced, cyclic, melu)
        self.save_dir       = f'outputs/{self.exp_type}_model' # we store the unlearnt model here
        self.retriever_model= 'paraphrase-MiniLM-L6-v2'
        self.forget_path    = './data/dpo_forget_idk.csv' 
        self.retain_path    = './data/full_retain_qa.csv'
        self.test_path      = './data/full_test_set.csv'
        self.melu_path      = './data/melu.csv'
        self.balanced_path   = './data/balanced_retain.csv'
        self.results_path   = f'/results/scores/{self.exp_type}_results.json'
        self.npo_beta       = 0.1
        self.npo_retain_alpha = 1.0
        self.npo_forget_gamma = 1.0




