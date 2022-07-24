import pandas as pd


class Preprocessor:
    '''
    text preprocessing
    '''

    def __init__(self) -> None:
        pass

    def read_file(self, file_path: str):
        df = pd.read_csv(file_path, skiprows=0, header=1)
        df = df.dropna()
        return df

    def prep_input_text(self, text):
        return f'Classification : {text}'

    def prep_target_text(self, text):
        return text

    def preprocess(self, data_paths: list, inputs_col: str, targets_col: str):
        '''
        :param data_paths: list of str
        '''
        df = pd.DataFrame()
        for data_path in data_paths:
            df = pd.concat([df, self.read_file(data_path)])

        df[inputs_col] = df[inputs_col].apply(self.prep_input_text)
        df[targets_col] = df[targets_col].apply(self.prep_target_text)

        return df

    def save_result(self, df, file_path):
        df.to_csv(file_path + '.csv', index=False, header=True)
        df.to_pickle(file_path + '.pkl')


if __name__ == '__main__':
    preprocessor = Preprocessor()
    res = preprocessor.preprocess(
        ['../data/SPAM.csv'],
        inputs_col='Message',
        targets_col='Category')
    preprocessor.save_result(res, '../data/SPAM_preprocessed')
