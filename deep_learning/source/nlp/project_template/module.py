import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)


class GenerateModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.save_hyperparameters()

        # self.hparams = hparams
        # self.model = hparams.model
        # self.tokenizer = hparams.tokenizer

        self.model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
        self.tokenizer = AutoTokenizer.from_pretrained(
            't5-small',
            model_max_length=self.hparams.max_length,
            is_fast=True
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != 'fit':
            return
        # Get dataloader by calling it - train_dataloader() is called after
        # setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * \
            float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size
        print('- setup module ... done')

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(
                        nd in n for nd in no_decay)], "weight_decay": self.hparams.weight_decay, }, {
                "params": [
                    p for n, p in model.named_parameters() if any(
                        nd in n for nd in no_decay)], "weight_decay": 0.0, }, ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1}
        print('- configure optimizer ... done')
        return [optimizer], [scheduler]
