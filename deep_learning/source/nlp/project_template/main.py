import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data_module import CustomDataModule
from module import GenerateModule


def setup_data_module(
    data_path,
    tokenizer,
    batch_size,
    max_length,
):
    df = pd.read_csv(data_path, skiprows=0, header=1)
    df = df.dropna()

    train_df = df.iloc[:int(len(df) * 0.8)]
    valid_df = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
    test_df = df.iloc[int(len(df) * 0.9):]

    inputs_col = 'Message'
    outputs_col = 'Category'

    # print(df)
    data_module_cfg = {
        'train': {
            'inputs': train_df[inputs_col].to_list(),
            'outputs': train_df[outputs_col].to_list()
        },
        'valid': {
            'inputs': valid_df[inputs_col].to_list(),
            'outputs': valid_df[outputs_col].to_list()
        },
        'test': {
            'inputs': test_df[inputs_col].to_list(),
            'outputs': test_df[outputs_col].to_list()
        },
        'batch_size': batch_size,
        'max_length': max_length,
        'tokenizer': tokenizer,
    }
    data_module = CustomDataModule(data_module_cfg)
    data_module.setup('fit')

    # for debug
    print(
        'dataloarder input_ids shape : ',
        next(
            iter(
                data_module.train_dataloader()))['input_ids'].shape)
    return data_module


def main():
    USE_GPU = torch.cuda.is_available()

    base_cfg = dict(
        model_name_or_path='t5-small',
        tokenizer_name_or_path='t5-small',
        batch_size=32,
        max_length=128,
        fp_16=True,
        model_save_path='./models/'
    )

    pl.seed_everything(42)

    tokenizer = AutoTokenizer.from_pretrained(
        base_cfg['tokenizer_name_or_path'],
        model_max_length=base_cfg['max_length'],
        is_fast=True
    )
    data_module = setup_data_module(
        data_path='data/SPAM.csv',
        tokenizer=tokenizer,
        batch_size=base_cfg['batch_size'],
        max_length=base_cfg['max_length'],
    )

    module_cfg = dict(
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
        train_batch_size=base_cfg['batch_size'],
        eval_batch_size=base_cfg['batch_size'],
        max_length=base_cfg['max_length'],

        # !if put model or tokenizer in the config, it will be freezed
        # !please initialize them in the module
        # model=model,
        # tokenizer=tokenizer
    )

    module_hparams = argparse.Namespace(**module_cfg)
    module = GenerateModule(module_hparams)

    train_cfg = dict(
        accumulate_grad_batches=module_cfg['gradient_accumulation_steps'],
        num_devices=-1,
        # find description of the "accelerator" parameters in the documentation :
        # https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html#distributed-modes
        strategy="dp",
        max_epochs=1
    )
    trainer = pl.Trainer(**train_cfg)
    print('start fit module')
    trainer.fit(module, data_module)

    module.tokenizer.save_pretrained(base_cfg['model_save_path'])
    module.model.save_pretrained(base_cfg['model_save_path'])


if __name__ == '__main__':
    main()
