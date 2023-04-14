# 读取数据
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MyDatasets(Dataset):
    def __init__(self, data_path, cls_name, pretrain_word2vect, train=1, seq_len=512, max_files_limit=1000):
        # data - 编码后的数据
        super(MyDatasets, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_word2vect)
        self.cls_name = cls_name
        self.n_cls = len(cls_name)
        self.labels = one_hot_encoder(cls_name)
        self.seq_len = seq_len
        self.max_files_limit = max_files_limit

        if train:
            self.data_path = os.path.join(data_path, 'train')
        else:
            self.data_path = os.path.join(data_path, 'test')

        self.data = self.read_files()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with open(self.data[index], encoding='utf-8') as f:
            label = self.data[index].split("\\")[-2]
            label = self.labels[label]
            lines = f.read()
            token = self.tokenizer.encode_plus(lines,
                                            add_special_tokens=True,        # 添加CLS和SEQ
                                            max_length=self.seq_len,        # 设定序列长度
                                            return_token_type_ids=False,    # 是否设置token类型掩码
                                            padding='max_length',           # 短于max_length的序列的填充至最长序列长度
                                            truncation=True,                # 长于max_length的序列裁剪至最长序列长度
                                            return_attention_mask=True,     # 返回注意力掩码
                                            return_tensors='pt'             # 返回的tensor类型，pt为pytorch的tensor，tf为tensorflow的tensor
            )

        return {'label': label, 'input_ids': token['input_ids'].flatten(), 'attention_mask': token['attention_mask']}

    def read_files(self):
        files_path = []
        for label in self.cls_name:
            class_path = os.path.join(self.data_path, label)
            for file in os.listdir(class_path):
                files_path.append(os.path.join(class_path, file))

        return files_path


def one_hot_encoder(cls_name):
    labels = {}
    if len(cls_name) > 2:
        for i, cls in enumerate(cls_name):
            label = []
            for j in range(len(cls_name)):
                if j == i:
                    label.append(1)
                else:
                    label.append(0)
            labels[cls] = torch.Tensor(label)
    else:
        for i, cls in enumerate(cls_name):
            labels[cls] = i
    return labels



if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\Transformer\orignal_Transformer\aclImdb'
    class_name = ['neg', 'pos', 'unsup']
    pretrained_word2vect = r'C:\Users\13632\Documents\Python_Scripts\Transformer\orignal_Transformer\bert-base-uncased'

    data = MyDatasets(data_path, class_name, pretrained_word2vect)
    print(data.__getitem__(1)['input_ids'].size())
    print(data.__getitem__(2)['input_ids'].size())
