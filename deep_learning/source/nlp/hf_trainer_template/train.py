import argparse
import json
import logging
import os
import sys

import transformers
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from utils.arguments import (DatasetMakerArguments, DataTrainingArguments,
                             ModelArguments)
from utils.ds_maker import DatasetMaker

# init logging ###############################################################
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
###############################################################################
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def main(model_args, dataset_maker_args, data_args, training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(
                os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Setup logging
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    # Set the verbosity to info of the Transformers logger (on main process
    # only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    # todo: customize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path)

    # Create or Load datasets
    # todo: customize the dataset
    dataset_maker = DatasetMaker(
        tokenizer=tokenizer,
        input_max_len=dataset_maker_args.input_max_len,
        target_max_len=dataset_maker_args.target_max_len
    )
    if dataset_maker_args.create_dataset:
        dataset_maker.make(
            data_paths=dataset_maker_args.data_paths,
            inputs_col=dataset_maker_args.inputs_col,
            targets_col=dataset_maker_args.targets_col,
            split_cfg=dataset_maker_args.split_cfg,
            save_path_prefix=dataset_maker_args.save_path_prefix,
        )

    train_dataset = dataset_maker.load_dataset(data_args.train_file)
    validation_dataset = dataset_maker.load_dataset(data_args.validation_file)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    print(f"Reading Config from : {args.config}")

    with open(args.config, 'r') as f:
        config = json.load(f)

    tag = config['tag']
    os.environ['WANDB_PROJECT'] = config['wandb_project']

    model_args = ModelArguments(**config['model_args'])
    dataset_maker_args = DatasetMakerArguments(**config['dataset_maker_args'])
    data_args = DataTrainingArguments(**config['data_args'])

    config['training_args']['output_dir'] += f'/{tag}'
    config['training_args']['run_name'] = tag
    training_args = TrainingArguments(**config['training_args'])

    # model_args = ModelArguments(
    #     model_name_or_path='t5-small',
    # )

    # dataset_maker_args = DatasetMakerArguments(
    #     create_dataset=True,
    #     input_max_len=128,
    #     target_max_len=16,
    #     data_paths=[
    #         './data/SPAM_preprocessed.pkl'],
    #     inputs_col='Message',
    #     targets_col='Category',
    #     save_path_prefix='./data/dataset')

    # data_args = DataTrainingArguments(
    #     train_file='./data/dataset/train.pkl',
    #     validation_file='./data/dataset/valid.pkl',
    # )

    # tag = '20220724_v1'
    # output_dir = f"./models/{tag}"

    # # trainer description :
    # # https://huggingface.co/docs/transformers/main_classes/trainer

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     overwrite_output_dir=True,

    #     do_train=True,
    #     do_eval=True,
    #     evaluation_strategy='steps',
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     num_train_epochs=10,

    #     logging_strategy='steps',
    #     logging_steps=100,
    #     logging_first_step=True,

    #     save_strategy='steps',
    #     save_steps=100,
    #     save_total_limit=10,

    #     report_to='wandb',
    #     run_name=tag,
    # )

    main(model_args, dataset_maker_args, data_args, training_args)
