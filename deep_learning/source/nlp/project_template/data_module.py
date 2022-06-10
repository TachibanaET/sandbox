import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    '''
    Custom Dataset class for PyTorch Lightning.
    '''

    def __init__(self, inputs, outputs, tokenizer, max_length):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        out = self.outputs[idx]

        enc = self.tokenizer.encode_plus(
            inp,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        dec = self.tokenizer.encode_plus(
            out,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # ! without squeeze() : shape = (batch_size, 1, seq_len) -> error
        return dict(
            # input_text=inp,
            input_ids=enc['input_ids'].squeeze(),
            attention_mask=enc['attention_mask'].squeeze(),
            # output_ids=dec['input_ids'],
            # output_mask=dec['attention_mask'],
            labels=dec['input_ids'].squeeze(),
            # output_mask=dec['attention_mask'],

        )


class CustomDataModule(pl.LightningDataModule):
    '''
    Custom Data Module class for PyTorch Lightning.
    '''

    def __init__(self, cfg: dict):
        super().__init__()
        self.train = cfg['train']
        self.valid = cfg['valid']
        self.test = cfg['test']
        self.batch_size = cfg['batch_size']
        self.max_length = cfg['max_length']
        self.tokenizer = cfg['tokenizer']

    def setup(self, stage=None):
        if stage != 'fit':
            return

        self.train_dataset = CustomDataset(
            self.train['inputs'],
            self.train['outputs'],
            self.tokenizer,
            self.max_length)

        self.valid_dataset = CustomDataset(
            self.valid['inputs'],
            self.valid['outputs'],
            self.tokenizer,
            self.max_length)

        self.test_dataset = CustomDataset(
            self.test['inputs'],
            self.test['outputs'],
            self.tokenizer,
            self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count())
