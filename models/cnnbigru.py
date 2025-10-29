import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        return x

class BPBiGRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.bigru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.bigru(x)
        out = self.dropout(out)
        return out

class CNNBiGRU(nn.Module):
    def __init__(self, in_channels=1, conv_channels=[32, 64, 128], gru_hidden=64, dense_hidden=32, pool_out_len=64):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, conv_channels[0]),
            ConvBlock(conv_channels[0], conv_channels[1]),
            ConvBlock(conv_channels[1], conv_channels[2]),
        )
        self.bigru1 = BPBiGRUBlock(conv_channels[2], gru_hidden)
        self.bigru2 = BPBiGRUBlock(gru_hidden * 2, gru_hidden)
        self.global_pool = nn.AdaptiveAvgPool1d(pool_out_len)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(gru_hidden * 2 * pool_out_len, dense_hidden)
        self.out_sbp = nn.Linear(dense_hidden, 1)
        self.out_dbp = nn.Linear(dense_hidden, 1)
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.bigru1(x)
        x = x.permute(0, 2, 1)
        x = self.bigru2(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        sbp = self.out_sbp(x)
        dbp = self.out_dbp(x)
        return sbp, dbp