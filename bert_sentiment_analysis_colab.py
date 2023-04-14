import os
import sys
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from torch.utils.data import DataLoader, Dataset
from dataloader import *

from sklearn.metrics import accuracy_score
from tqdm import tqdm

data_path = r'/content/drive/MyDrive/Colab Notebooks/bert_sentiment_analysis/aclImdb'
class_name = ['neg', 'pos', 'unsup']
pretrained_word2vect = '/content/bert-base-uncased'
pretrained_model = '/content/bert-base-uncased'
save_apth = '/content/drive/MyDrive/Colab Notebooks/bert_sentiment_analysis'
epochs = 100

# Define the tokenizer and model
if not os.path.exists(pretrained_model):
    pretrained_model = 'bert-base-uncased'
    pretrained_word2vect = 'bert-base-uncased'
    print('Downloading pretained from web...')
else:
    print('Loading pretrained model from {}...'.format(pretrained_model))
# 此模型预设了损失函数，回归问题使用均方差，二分类使用交叉熵，多分类使用BCEWithLogitsLoss
model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=len(class_name))

# Define the training and testing datasets
train_dataset = MyDatasets(data_path=data_path, cls_name=class_name, pretrain_word2vect=pretrained_word2vect)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The model will run in {}'.format(device))
model.to(device)


for epoch in range(epochs):
    train_loss = 0.0
    train_acc = 0.0

    model.train()
    with tqdm(total=len(train_loader)) as pbar:
        for batch in train_loader:
            pbar.set_description('epoch - {}'.format(epoch+1))

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            # outputs是一个类，包含损失，分类概率，hidden_state，attention(这两个啥？
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            logits = outputs.logits

            pbar.set_postfix('loss: {}'.format(loss.item))

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            train_acc += accuracy_score(labels.cpu(), preds.cpu())

            loss.backward()
            optimizer.step()

            pbar.update(1)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # print(f'Epoch {epoch + 1}/{3}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

torch.save(model.state_dict(), save_apth)


def setting(num_attention_heads, num_hidden_layers):
    config_path = r'C:\Users\13632\Documents\Python_Scripts\Transformer\orignal_Transformer\bert-base-uncased\config.json'
    with open(config_path, 'rb') as f:
        # 定义为只读模型，并定义名称为f
        params = json.load(f)
        # 将修改后的内容保存在dict中
        params['num_attention_heads'] = num_attention_heads
        params['num_hidden_layers'] = num_hidden_layers
        dict = params

    with open(config_path, 'w') as f:
        json.dump(dict, f)

    print('Successfully change params in config.')
