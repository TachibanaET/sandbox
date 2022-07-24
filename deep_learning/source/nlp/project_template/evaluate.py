import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TextGenerator:
    def __init__(
            self,
            model,
            tokenizer,
            max_length,
            device):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __format_prompt(self, prompt):
        return f'Classification: {prompt}'
        # return prompt

    def generate(self, prompt):
        prompt = self.__format_prompt(prompt)
        print(f'prompt :> {prompt}')
        tokenized_prompt = self.tokenizer.batch_encode_plus(
            [prompt],
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        input_ids = tokenized_prompt["input_ids"]
        attention_mask = tokenized_prompt['attention_mask']

        # debug
        print(f'input_ids :> {input_ids}')
        print(f'attention_mask :> {attention_mask}')

        if self.device != 'cpu':
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            self.model = self.model.to(self.device)

        outs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            return_dict_in_generate=True,
            output_scores=True)

        print(outs)
        dec = [self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
            for ids in outs.sequences][0]
        return dec


if __name__ == '__main__':
    model_path = './models/20220723_test_v1'
    # model_path = 't5-small'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_length = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_generator = TextGenerator(model, tokenizer, max_length, device)

    while(1):
        prompt = input('> ')
        print(text_generator.generate(prompt))
