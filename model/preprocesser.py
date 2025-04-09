import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
class SentencePairDataset(Dataset):
    def __init__(self, data, bert_tokenizer, codebert_tokenizer,message='message',command='command',label='label'):
        self.data = data
        self.message = message
        self.command = command
        self.label = label
        self.bert_tokenizer = bert_tokenizer
        self.codebert_tokenizer = codebert_tokenizer

    def __len__(self):
        return len(self.data)

    def get_embedding(self, idx):
        # print(self.data.loc[idx][message], self.data.loc[idx][command],self.data.loc[idx]['label'])
        sentence1, sentence2, label = str(self.data.loc[idx][self.message]), str(self.data.loc[idx][self.command]), int(self.data.loc[idx][self.label])
        inputs_bert = self.bert_tokenizer(sentence1, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        inputs_codebert = self.codebert_tokenizer(sentence2, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        return {k: v.squeeze(0) for k, v in inputs_bert.items()}, {k: v.squeeze(0) for k, v in inputs_codebert.items()}, label
        # this following line was basic though, anyway it fails
        # return inputs_bert, inputs_codebert, label

    def __getitem__(self, idx):
        return self.get_embedding(idx)
    # def __getitem__(self, index):
    #     # 直接返回原始数据而不做 tokenization
    #     message = self.data.iloc[index][self.message]
    #     code = self.data.iloc[index][self.command]
    #     label = self.data.iloc[index][self.label]
    #     return message, code, label
