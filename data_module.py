from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import PreTrainedTokenizer, default_data_collator
from typing import Tuple
import math
import pandas as pd
from typing import Dict, List, Set, Tuple, Any



def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    full_text = question + answer
    num_question_tokens = len(tokenizer.tokenize(question, add_special_tokens=False)) #this is important, we 
    encoded = tokenizer(
        full_text,
        add_special_tokens=False, #this is important, we keep false cause we already added the special tokens from template
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    #change label to -100 for question tokens, including assistant header and end of header.
    for i in range(num_question_tokens): label[i] = -100
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class SingleDataset(Dataset):
    def __init__(self, forget_data,
                 tokenizer,
                 max_length=512,
                 question_key = 'question',
                 answer_key = 'answer'):
        """
        Initializes the dataset for gradient ascent finetuning

        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            max_length (int, optional): maximum sequence length for tokenization. Defaults to 512.
            template_format (str, optional): format template for structuring input
        """
        self.data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx][self.qk]
        answer = self.data.iloc[idx][self.ak]
        return convert_raw_data_to_model_qa(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            question=question,
            answer=answer
        )


class DualDatasetRandom(Dataset):
    """
    TOFU way of implementation.

    Args:
        forget_data (pd.DataFrame): DataFrame for forgetting.
        retain_data (pd.DataFrame): DataFrame for retaining.
        tokenizer: tokenizer instance to process text.
        max_length (int): maximum sequence length.
        question_key (str): column name for questions.
        answer_key (str): column name for answers.
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        forget_idx = idx
        # A new random sample is chosen every time __getitem__ is called.
        retain_idx = torch.randint(0, len(self.retain), (1,)).item()

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)
    

class DualDataset(Dataset):
    """
    Dataset class for creating data for forget and retain (used by gradient difference)

    Args:
        forget_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for forgetting
        retain_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for retaining
        tokenizer: tokenizer instance to process text
        max_length (int): maximum sequence length
        template_format (str, optional): format template for structuring input

    Returns:
        Tuple of forget and retain samples:
        (
            (forget_input_ids, forget_labels, forget_attention_mask),
            (retain_input_ids, retain_labels, retain_attention_mask)
        )
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
    def __len__(self):
        return max(len(self.forget), len(self.retain))

    def __getitem__(self, idx):
        # Cyclic rotation of data
        forget_idx = idx % len(self.forget)
        retain_idx = idx % len(self.retain)

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)



class DualTitleDataset(Dataset):
    """
    Dataset that returns pre-paired forget/retain rows.

    Expects a DataFrame with columns:
      question_forget, answer_forget, question_retain, answer_retain
    (plus any other columns you don’t care about).
    """
    def __init__(
        self,
        paired_df,
        tokenizer,
        max_length,
        question_key: str = "question",
        answer_key: str = "answer"
    ):
        # e.g. paired_df.columns = [
        #   'title', 'question_forget', 'answer_forget',
        #   'question_retain', 'answer_retain', … ]
        self.df = paired_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # pull out the two sides
        q_forget = row[f"{self.qk}_forget"]
        a_forget = row[f"{self.ak}_forget"]
        q_retain = row[f"{self.qk}_retain"]
        a_retain = row[f"{self.ak}_retain"]

        # convert as before
        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q_forget, a_forget
        )
        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q_retain, a_retain
        )

        return (forget_data, retain_data)

