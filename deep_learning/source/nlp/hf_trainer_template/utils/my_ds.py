from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, inputs: list, targets: list):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return dict(
            input_ids=self.inputs[idx]["input_ids"].squeeze(),
            attention_mask=self.inputs[idx]["attention_mask"].squeeze(),
            decoder_attention_mask=self.targets[idx]['attention_mask'].squeeze(),
            labels=self.targets[idx]['input_ids'].squeeze(),
        )

    def show_example(self):
        idx = 0
        print('----- show example -----')
        print('input_ids : ', self.inputs[idx]["input_ids"])
        print('attention_mask : ', self.inputs[idx]["attention_mask"])
        print('decoder_attention_mask : ', self.targets[idx]['attention_mask'])
        print('labels : ', self.targets[idx]['input_ids'])
