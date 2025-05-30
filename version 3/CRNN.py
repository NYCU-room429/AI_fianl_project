import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score


class CRNN(nn.Module):
    def __init__(self, num_classes=129):
        super(CRNN, self).__init__()

        self.pad = nn.ZeroPad2d((37, 37, 0, 0))
        self.bn0 = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop1 = nn.Dropout(0.5)  # dropout 機率調高

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop3 = nn.Dropout(0.5)

        self.gru1 = nn.GRU(
            input_size=128, hidden_size=64, batch_first=True, bidirectional=True
        )
        self.gru2 = nn.GRU(
            input_size=128, hidden_size=64, batch_first=True, bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=2, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(128)
        self.attn_dropout = nn.Dropout(0.5)

        self.drop_final = nn.Dropout(0.7)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        x = self.pad(x)
        x = self.bn0(x)

        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.mean(dim=2)  # (batch, channels, pooled_time)
        x = x.permute(0, 2, 1)  # (batch, pooled_time, channels)

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = self.attn_norm(x + attn_output)
        x = self.attn_dropout(x)

        x = self.drop_final(x)
        x = self.fc(x)  # (batch, pooled_time, num_classes)

        if x.shape[1] != 1000:
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, size=1000, mode="linear", align_corners=False)
            x = x.permute(0, 2, 1)
        return x


def train(model, train_loader, criterion, optimizer, device):
    total_loss = 0
    model.train()
    for mel, label in tqdm(train_loader, desc="Training"):
        mel = mel.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, val_loader, criterion, device, threshold=0.5):
    total_loss = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc="Validation"):
            mel = mel.to(device)
            label = label.to(device)
            outputs = model(mel)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    avg_loss = total_loss / len(val_loader)
    all_outputs = np.concatenate(all_outputs, axis=0).reshape(-1, outputs.shape[-1])
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1, outputs.shape[-1])
    return avg_loss, all_outputs, all_labels
