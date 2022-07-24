import os
import pickle

import pandas as pd
from tqdm import tqdm

from utils.my_ds import MyDataset


class DatasetMaker:
    def __init__(self, tokenizer, input_max_len, target_max_len) -> None:
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len

    def read_file(self, file_path: str):
        return pd.read_pickle(file_path)

    def create_dataset(self, text_inputs: list, text_targets: list):
        inputs = []
        targets = []
        for i, t in tqdm(
                zip(text_inputs, text_targets),
                total=len(text_inputs)):
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [i], max_length=self.input_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [t], max_length=self.target_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            inputs.append(tokenized_inputs)
            targets.append(tokenized_targets)

        return MyDataset(inputs, targets)

    def save_dataset(self, dataset, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)

    def load_dataset(self, file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def make(
            self,
            data_paths: list,
            inputs_col: str,
            targets_col: str,
            split_cfg: dict,
            save_path_prefix: str):
        '''
        :param data_paths: list of str
        :param inputs_col: str
        :param targets_col: str
        :param split_cfg: {'train': float, 'valid': float, 'test': float} (train + valid + test = 1)
        '''
        df = pd.DataFrame()
        for data_path in data_paths:
            df = df.append(self.read_file(data_path))

        data_size = df.shape[0]
        train_size = int(data_size * split_cfg['train'])
        valid_size = int(data_size * split_cfg['valid'])

        train_df = df.iloc[:train_size]
        valid_df = df.iloc[train_size:train_size + valid_size]
        test_df = df.iloc[train_size + valid_size:]

        os.makedirs(save_path_prefix, exist_ok=True)

        if not train_df.empty:
            train_dataset = self.create_dataset(
                train_df[inputs_col], train_df[targets_col])
            self.save_dataset(
                train_dataset,
                f'{save_path_prefix}/train.pkl')

        if not valid_df.empty:
            valid_dataset = self.create_dataset(
                valid_df[inputs_col], valid_df[targets_col])
            self.save_dataset(
                valid_dataset,
                f'{save_path_prefix}/valid.pkl')

        if not test_df.empty:
            test_dataset = self.create_dataset(
                test_df[inputs_col], test_df[targets_col])
            self.save_dataset(
                test_dataset,
                f'{save_path_prefix}/test.pkl')

        print('DatasetMaker: make() done')
        print(
            f'train: {train_df.shape}, valid: {valid_df.shape}, test: {test_df.shape}')

        train_dataset.show_example()


if __name__ == '__main__':
    # example
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    input_max_len = 128
    target_max_len = 16

    split_cfg = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    ds_maker = DatasetMaker(
        tokenizer=tokenizer,
        input_max_len=input_max_len,
        target_max_len=target_max_len)

    ds_maker.make(
        ['../data/SPAM_preprocessed.pkl'],
        inputs_col='Message',
        targets_col='Category',
        split_cfg=split_cfg,
        save_path_prefix='../data/dataset')
