
train = True

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange
from statistics import mean
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import trange, tqdm
from sklearn.metrics import accuracy_score, classification_report
import os
class Embedding(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self._pretrained = pretrained
        self._config = AutoConfig.from_pretrained(self._pretrained)
        self.bert = AutoModel.from_pretrained(self._pretrained, config=self._config)

    def _freeze(self):
        print('Unfreezing the last layer of BERT...')
        for params in self.bert.parameters():
            params.requires_grad = False

        if 'distil' in self._pretrained.lower():
            print('Distillation...')

        else:
            for params in self.bert.encoder.layer[-1].parameters():
                params.requires_grad = True
        print('Unfreezed the last layer successfully.')


    def forward(self, encoded_dict):
        output = self.bert(**encoded_dict)
        return output.last_hidden_state[:, 0, :]

class Classifier(nn.Module):
    def __init__(self, input_channels=768):
        super().__init__()
        self.lin = nn.Linear(input_channels, 3)

    def forward(self, x):
        lin = self.lin(x)
        output = lin
        return output

class Model(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.encode = Embedding(pretrained)
        self.encode._freeze()
        self.decode = Classifier()

    def forward(self, encoded_dict):
        emb = self.encode(encoded_dict)
        output = self.decode(emb)
        return output

class BertDataset(Dataset):
    def __init__(self, dataframe, pretrained, train=True):
        self._dataframe = dataframe
        self._train = train
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def _preprocess(self, text):
        encoded_dict = self._tokenizer.encode_plus(text=text, add_special_tokens=True, max_length=512, padding='max_length',
                                                   truncation=True, return_attention_mask=True, return_tensors='pt')
        return encoded_dict

    def __getitem__(self, index):
        encoded_dict = self._preprocess(self._dataframe.iloc[index]['feature'])
        for el in encoded_dict:
            encoded_dict[el] = encoded_dict[el].squeeze()
        encoded_dict['label'] = torch.tensor(self._dataframe.iloc[index]['class'])
        return encoded_dict

    def __len__(self):
        return len(self._dataframe)

class TestDataset(Dataset):
    def __init__(self, dataframe, pretrained):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        encoded_dict = self._preprocess(self.dataframe.iloc[index]['feature'])
        for el in encoded_dict:
            encoded_dict[el] = encoded_dict[el].squeeze()
        encoded_dict['label'] = torch.tensor(self.dataframe.iloc[index]['class'])
        return encoded_dict

    def _preprocess(self, text):
        encoded_dict = self.tokenizer.encode_plus(text=text, add_special_tokens=True, max_length=512,
                                                  padding='max_length', truncation=True, return_attention_mask=True,
                                                  return_tensors='pt')
        return encoded_dict

class Main():
    def __init__(self, device, pretrained, model_file=None):
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        self.pretrained = pretrained
        self.model = Model(self.pretrained)
        self.model.to(self.device)
        if model_file is not None:
            if os.path.exists(model_file):
                self.model.load_state_dict(torch.load(model_file, map_location = self.device))

    def prepare_dataloader(self, data, train_ratio=0.8, val_ratio=0.1, batchsize=8, test_dataset_path=None):
        dataset = BertDataset(data, self.pretrained)
        size = len(data)
        train_size = int(train_ratio * size)
        val_size = int(val_ratio * size)
        test_size = size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        if test_dataset_path is not None:
            test_dataset.dataset._dataframe.to_pickle(test_dataset_path)

        train_loader = DataLoader(train_dataset, batch_size=batchsize)
        val_loader = DataLoader(val_dataset, batch_size=batchsize)
        test_loader = DataLoader(test_dataset, batch_size=batchsize)

        return train_loader, val_loader, test_loader

    def evaluate(self, dataloader):
        total_loss = 0
        all_predictions = []
        all_ground_truth = []

        for data in dataloader:
            data = {key: val.to(self.device) for key, val in data.items()}
            encoded_dict = {key: data[key] for key in data.keys() if key != 'label'}
            y = data['label'].to(torch.long)
            y_pred = self.model(encoded_dict)
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()

            predictions = torch.argmax(y_pred, dim=1).cpu().numpy()
            ground_truth = y.cpu().numpy()
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)

        accuracy = accuracy_score(all_ground_truth, all_predictions)
        return total_loss / len(dataloader), accuracy

    def train(self, train_loader, val_loader, test_loader, epochs, model_file, opt=None, lr_scheduler=None):
        best_dev_loss = sys.maxsize
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_epoch = -99

        print('Training starting...')
        for epoch in trange(epochs):
            self.model.train()
            loss_val = []
            avg_loss_epoch = []
            if epoch == 0:
                train_loss, train_accuracy = self.evaluate(train_loader)
                val_loss, val_accuracy = self.evaluate(val_loader)
                print(
                    f'Epoch {epoch} / {epochs}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.4f}')
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)

            start = len(val_losses)
            for data in train_loader:
                data = {key: val.to(self.device) for key, val in data.items()}
                encoded_dict = {key: data[key] for key in data.keys() if key != 'label'}
                y = data['label'].to(torch.long)
                y_pred = self.model(encoded_dict)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                opt.step()
                loss_val.append(loss.item())

            end = len(val_losses)
            avg_loss_epoch.append(mean(loss_val[start: end + 1]))

            with torch.no_grad():
                self.model.eval()
                train_loss, train_accuracy = self.evaluate(train_loader)
                val_loss, val_accuracy = self.evaluate(val_loader)

                if val_loss < best_dev_loss:
                    best_dev_loss = val_loss
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), model_file)

                if lr_scheduler is not None:
                    lr_scheduler.step(val_loss)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)

                print(
                    f'Epoch {epoch} / {epochs}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.4f}')

            if epoch - best_epoch > 5:
                print('Early stopping....')
                break

        print(
            f'Epoch {epoch} / {epochs}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.4f}, Test Loss: {self.evaluate(test_loader)[0]:.6f}')

    def predict(self, dataloader):
        total_loss = 0
        self.model.eval()
        self.model.to(self.device)
        y_class = []
        ground_truth = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = {key: val.to(self.device) for key, val in data.items()}
                encoded_dict = {key: data[key] for key in data.keys() if key != 'label'}
                y = data['label'].to(torch.long)
                y_pred = self.model(encoded_dict)
                y_pred_prob = torch.softmax(y_pred, dim=1)
                y_class.extend(torch.argmax(y_pred_prob, dim=1).detach().cpu().numpy())
                ground_truth.extend(y.detach().cpu().numpy())
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()
        print('Test loss is %.6f' % (total_loss / len(dataloader)))
        print('Classification Report')
        print(classification_report(ground_truth, y_class))

def run_cls(dataset, device, pretrained, lr=1e-5, weight_decay=0.001,
            epochs=30, model_file=None, train=True, train_file_path=None, test_file_path=None):

    mainclass = Main(device, pretrained, model_file=model_file)

    if train:
        train_loader, val_loader, test_loader = mainclass.prepare_dataloader(dataset, test_dataset_path=test_file_path)

        opt = optim.Adam(mainclass.model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
        mainclass.train(train_loader, val_loader, test_loader, epochs, model_file, opt=opt, lr_scheduler=lr_scheduler)
    else:
        test_dataset = pd.read_pickle(test_file_path)
        custom_test_dataset = TestDataset(test_dataset, pretrained)
        test_loader = DataLoader(custom_test_dataset, batch_size=8, shuffle=False)

        mainclass.model_file = model_file
        mainclass.predict(test_loader)