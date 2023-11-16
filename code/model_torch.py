import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chardet

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as rawdata:
            result = chardet.detect(rawdata.read(10000))
            print(result)
        df = pd.read_csv(file_path,encoding=result['encoding'])
        self.x = df.iloc[:, 2:-1].values
        self.y = df.iloc[:, -1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] ** 2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y

    def __len__(self):
        return self.length
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x
    
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('USING PYTORCH VERSION :', torch.__version__, 'DEVICE :',DEVICE)

BATCH_SIZE = 32
EPOCHS = 10
data_df = CustomDataset("list_100.xlsx")
size = len(data_df)
train_dataset, validation_dataset, test_dataset = random_split(data_df, [int(size*0.8),int(size*0.1),int(size*0.1)])

print(len(train_dataset))
print(len(validation_dataset))
print(len(test_dataset))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = CustomModel().to(DEVICE)
criterion = nn.MSELoss().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_dataloader)

with torch.no_grad():
    model.eval()
    for x, y in validation_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        outputs = model(x)
        print(f"X : {x}")
        print(f"Y : {y}")
        print(f"Outputs : {outputs}")
        print("--------------------")