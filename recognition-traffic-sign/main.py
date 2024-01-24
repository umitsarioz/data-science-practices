import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 200)

root_fp = os.path.join(os.getcwd(), "recognition-traffic-sign", "dataset")
train_meta_fp = os.path.join(root_fp, "Train.csv")
test_meta_fp = os.path.join(root_fp, "Test.csv")

df_train_meta = pd.read_csv(train_meta_fp)
df_test_meta = pd.read_csv(test_meta_fp)

df_train_meta.head()
df_test_meta.head()

# Check missing values
df_train_meta.isna().any()
df_test_meta.isna().any()


# convert labels to one hot array
def to_categorical(val, n_label) -> list:
    arr = np.zeros(n_label)
    arr[val] = 1
    return arr


def create_labels(df_train, df_test) -> tuple:
    labels = df_train_meta["ClassId"].sort_values().unique().tolist()
    train_labels = [to_categorical(val=val, n_label=len(labels)) for val in df_train["ClassId"].values]
    test_labels = [to_categorical(val=val, n_label=len(labels)) for val in df_test["ClassId"].values]
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    return train_labels, test_labels


def to_pixel(image_path):
    height, width = 32, 32
    dim = (height, width)
    img = Image.open(image_path)  # load
    img = img.resize(dim)  # resize
    img = np.asarray(img) / 255.  # normalization
    return img


def create_features(df_train, df_test) -> tuple:
    train_images_fp = [os.path.join(root_fp, img_path) for img_path in df_train["Path"].values]
    test_images_fp = [os.path.join(root_fp, img_path) for img_path in df_test["Path"].values]
    train_pixels = [to_pixel(image_path=img_path) for img_path in train_images_fp]  # get pixel values
    test_pixels = [to_pixel(image_path=img_path) for img_path in test_images_fp]  # get pixel values
    train_pixels = np.asarray(train_pixels)
    test_pixels = np.asarray(test_pixels)
    return train_pixels, test_pixels


train_labels, test_labels = create_labels(df_train=df_train_meta, df_test=df_test_meta)
train_features, test_features = create_features(df_train=df_train_meta, df_test=df_test_meta)


# Split training data as validation and training

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=.1, random_state=17,
                                                  shuffle=True)
X_test, y_test = test_features, test_labels

X_train_torch, y_train_torch = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
X_val_torch,  y_val_torch = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
X_test_torch, y_test_torch = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

# From: [batch_size, depth, height, width, channels]
# To: [batch_size, channels, depth, height, width]
X_train_torch.size()
y_train_torch.size()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train_torch, y_train_torch = X_train_torch.permute(0,3,1,2).cuda(device), y_train_torch.permute(0,1).cuda(device)
X_val_torch, y_val_torch = X_val_torch.permute(0,3,1,2).cuda(device), y_val_torch.permute(0,1).cuda(device)
X_test_torch, y_test_torch = X_test_torch.permute(0,3,1,2).cuda(device), y_test_torch.permute(0,1).cuda(device)

X_train_torch.size()
X_val_torch.size()

# There is bug on model.I need to examine calculating logic of cnn again, there are some dimensionality problems ..
# Alternatively you can examine this repo for same project is developed with tensorflow keras: https://github.com/umitsarioz/traffic-sign-notifier

class Net(nn.Module):
    def __init__(self, n_label: int, n_channel: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=.2)
        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1)
        self.fc = nn.Linear(in_features=8 * 16 * 16, out_features=n_label)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = f.softmax(self.fc(x))
        return x


model = Net(n_label=43, n_channel=3).to(device=device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
