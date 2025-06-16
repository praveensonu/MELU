import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import pandas as pd
import math
from data_module import convert_raw_data_to_model_qa


class VanillaDPODataset(Dataset):

    def __init__(self, forget_data: pd.DataFrame, tokenizer: Any,
                 max_length: int,
                 question_key: str = 'question',
                 answer_key: str = 'answer',
                 idk_key: str = 'idk'):
        if not all(k in forget_data.columns for k in [question_key, answer_key, idk_key]):
             raise ValueError(f"forget_data must contain columns: {question_key}, {answer_key}, {idk_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
        self.ik = idk_key

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        row = self.forget_data.iloc[idx]
        q = row[self.qk]
        ans = row[self.ak]
        idk = row[self.ik]

        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, ans,
                                                )
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, idk,
                                                )

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
        }


class ForgetIdkRetainDataset(Dataset):
    """
    For each row in forget_data (must have 'question','answer','idk') and the
    parallel retain_data (must have 'question','answer'), returns a dict:
      {
        'answer_input_ids': …,
        'answer_labels': …,
        'answer_attention_mask': …,
        'idk_input_ids': …,
        'idk_labels': …,
        'idk_attention_mask': …,
        'retain_input_ids': …,
        'retain_labels': …,
        'retain_attention_mask': …,
      }

    Basically, for each sample, it return a dictionary of forget + idk and retain inputs.
    """
    def __init__(
        self,
        forget_data: pd.DataFrame,
        retain_data: pd.DataFrame,
        tokenizer,
        max_length: int,
        question_key: str = 'question',
        answer_key: str = 'answer',
        idk_key: str = 'idk',
    ):
        # validate
        if not all(col in forget_data.columns for col in [question_key, answer_key, idk_key]):
            raise ValueError(f"forget_data must contain: {question_key}, {answer_key}, {idk_key}")
        if not all(col in retain_data.columns for col in [question_key, answer_key]):
            raise ValueError(f"retain_data must contain: {question_key}, {answer_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk, self.ak, self.ik = question_key, answer_key, idk_key

    def __len__(self):
        # assumes forget_data and retain_data are same length
        return len(self.forget_data)

    def __getitem__(self, idx):
        f_row = self.forget_data.iloc[idx]
        r_row = self.retain_data.iloc[idx]

        # forget answer
        q = f_row[self.qk]
        ans = f_row[self.ak]
        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, ans)

        # forget "idk"
        idk = f_row[self.ik]
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, idk)

        # retain answer
        retain_q = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, retain_q, retain_ans)

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
            'retain_input_ids':      ri,
            'retain_labels':         rl,
            'retain_attention_mask': rm,
        }


class ForgetIdkRetainDatasetRandom(Dataset):
    """
    For each row in forget_data, returns a dictionary containing three items:
    1. The forget question paired with its original answer.
    2. The forget question paired with its "I don't know" answer.
    3. A RANDOMLY selected question-answer pair from the retain_data.

    Output format is a dictionary of tensors:
      {
        'answer_input_ids': ..., 'answer_labels': ..., 'answer_attention_mask': ...,
        'idk_input_ids': ..., 'idk_labels': ..., 'idk_attention_mask': ...,
        'retain_input_ids': ..., 'retain_labels': ..., 'retain_attention_mask': ...,
      }
    """
    def __init__(
        self,
        forget_data: pd.DataFrame,
        retain_data: pd.DataFrame,
        tokenizer,
        max_length: int,
        question_key: str = 'question',
        answer_key: str = 'answer',
        idk_key: str = 'idk',
    ):
        # validate
        if not all(col in forget_data.columns for col in [question_key, answer_key, idk_key]):
            raise ValueError(f"forget_data must contain: {question_key}, {answer_key}, {idk_key}")
        if not all(col in retain_data.columns for col in [question_key, answer_key]):
            raise ValueError(f"retain_data must contain: {question_key}, {answer_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk, self.ak, self.ik = question_key, answer_key, idk_key

    def __len__(self):
        # The length of an epoch is determined by the number of samples to forget.
        return len(self.forget_data)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        f_row = self.forget_data.iloc[idx]

        # CHANGED: The retain sample is chosen RANDOMLY from the entire retain set.
        random_retain_idx = torch.randint(0, len(self.retain_data), (1,)).item()
        r_row = self.retain_data.iloc[random_retain_idx]

        # --- The rest of the logic remains the same ---

        # Process forget sample with its original answer
        q = f_row[self.qk]
        ans = f_row[self.ak]
        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, ans)

        # Process forget sample with its "idk" answer
        idk = f_row[self.ik]
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, idk)

        # Process the RANDOMLY CHOSEN retain sample
        retain_q = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, retain_q, retain_ans)

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
            'retain_input_ids':      ri,
            'retain_labels':         rl,
            'retain_attention_mask': rm,
        }
    
    

class MELUForgetIdkRetainDataset(Dataset):
    """
    Expects a single DataFrame with columns:
      question_forget, answer_forget, idk_forget,
      question_retain, answer_retain

    Returns, for each row, a dict with tokenized inputs/labels/masks for:
      - forget-answer
      - forget-idk
      - retain-answer
    """
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int,
        # Override if column names differ:
        q_forget_key: str = 'question_forget',
        a_forget_key: str = 'answer_forget',
        idk_forget_key: str = 'idk_forget',
        q_retain_key: str = 'question_retain',
        a_retain_key: str = 'answer_retain',
    ):
        required = [
            q_forget_key, a_forget_key, idk_forget_key,
            q_retain_key, a_retain_key
        ]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # store the keys
        self.qf, self.af, self.ifk = q_forget_key, a_forget_key, idk_forget_key
        self.qr, self.ar = q_retain_key, a_retain_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---- forget-answer ----
        qf = row[self.qf]
        af = row[self.af]
        fa_input_ids, fa_labels, fa_attention_mask = \
            convert_raw_data_to_model_qa(self.tokenizer, self.max_length, qf, af)

        # ---- forget-idk ----
        idkf = row[self.ifk]
        fi_input_ids, fi_labels, fi_attention_mask = \
            convert_raw_data_to_model_qa(self.tokenizer, self.max_length, qf, idkf)

        # ---- retain-answer ----
        qr = row[self.qr]
        ar = row[self.ar]
        ra_input_ids, ra_labels, ra_attention_mask = \
            convert_raw_data_to_model_qa(self.tokenizer, self.max_length, qr, ar)

        return {
            # forget-answer
            'answer_input_ids':      fa_input_ids,
            'answer_labels':         fa_labels,
            'answer_attention_mask': fa_attention_mask,
            # forget-idk
            'idk_input_ids':         fi_input_ids,
            'idk_labels':            fi_labels,
            'idk_attention_mask':    fi_attention_mask,
            # retain-answer
            'retain_input_ids':      ra_input_ids,
            'retain_labels':         ra_labels,
            'retain_attention_mask': ra_attention_mask,
        }



class CyclicForgetIdkRetainDataset(Dataset):
    """
    Cycles through the *shorter* split so that every row of the *longer*
    split is visited exactly once per epoch.  In the common case where
    retain_data is larger, you iterate over retain_data sequentially and
    wrap around forget_data via idx % len(forget_data).
    """
    def __init__(
        self,
        forget_data: pd.DataFrame,
        retain_data: pd.DataFrame,
        tokenizer,
        max_length: int,
        question_key: str = 'question',
        answer_key: str = 'answer',
        idk_key: str = 'idk',
    ):
        # validation
        req_f = {question_key, answer_key, idk_key}
        req_r = {question_key, answer_key}
        if not req_f.issubset(forget_data.columns):
            raise ValueError(f"forget_data must contain: {', '.join(req_f)}")
        if not req_r.issubset(retain_data.columns):
            raise ValueError(f"retain_data must contain: {', '.join(req_r)}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk, self.ak, self.ik = question_key, answer_key, idk_key
        self.f_len = len(self.forget_data)
        self.r_len = len(self.retain_data)

    def __len__(self):
        """Length is the *longer* split so that we see every row once."""
        return max(self.f_len, self.r_len)

    def _row(self, df, idx, modulo_len):
        """Helper to get a row with modulo wrap-around."""
        return df.iloc[idx % modulo_len]

    def __getitem__(self, idx):
        f_row = self._row(self.forget_data, idx, self.f_len)
        r_row = self._row(self.retain_data, idx, self.r_len)

        q = f_row[self.qk]
        ans = f_row[self.ak]
        ai, al, am = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q, ans
        )

        idk = f_row[self.ik]
        ii, il, im = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q, idk
        )

        retain_q   = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, retain_q, retain_ans
        )

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
            'retain_input_ids':      ri,
            'retain_labels':         rl,
            'retain_attention_mask': rm,
        }



