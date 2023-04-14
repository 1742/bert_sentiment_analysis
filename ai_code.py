import sys

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from torch.utils.data import DataLoader, Dataset
from dataloader import *

from sklearn.metrics import accuracy_score
from tqdm import tqdm

data_path = r'C:\Users\13632\Documents\Python_Scripts\Transformer\orignal_Transformer\aclImdb'
class_name = ['neg', 'pos']
pretrained_word2vect = r'C:\Users\13632\Documents\Python_Scripts\Transformer\orignal_Transformer\bert-base-uncased'
pretrained_model = r'C:\Users\13632\Documents\Python_Scripts\Transformer\orignal_Transformer\bert-base-uncased'
epochs = 10

# Define the tokenizer and model
pretrained_word2vect = BertTokenizer.from_pretrained(pretrained_word2vect)
# 此模型预设了损失函数，回归问题使用均方差，二分类使用交叉熵，多分类使用BCEWithLogitsLoss
# 模型参数可以在C:\Users\13632\Documents\Python_Scripts\Transformer\orignal_Transformer\bert-base-uncased\config.json中查看更改
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

            pbar.set_postfix({'CELoss:': loss.item()})

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            train_acc += accuracy_score(labels.cpu(), preds.cpu())

            loss.backward()
            optimizer.step()

            pbar.update(1)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # print(f'Epoch {epoch + 1}/{3}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
