import torch
import torch.nn as nn
import torch.nn.functional as F


# === 1. 通道注意力模块 ===
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# === 2. 空间注意力模块 ===
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# === 3. CBAM ===
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# === 主网络 ===
class PhysicsGuidedNet(nn.Module):
    def __init__(self, num_rx=6, signal_len=2048, map_size=512):
        super(PhysicsGuidedNet, self).__init__()

        # --- A. IQ 分支 ---
        self.iq_encoder = nn.Sequential(
            nn.Conv1d(num_rx * 2, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, 2, 1), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )

        # --- B. Heatmap 分支 ---
        self.enc1 = self._conv_block(1, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        self.enc5 = self._conv_block(128, 256)
        self.enc6 = self._conv_block(256, 512)

        self.map_attention = CBAM(512)

        # --- C. 融合与回归 ---
        class Mish(nn.Module):
            def forward(self, x):
                return x * torch.tanh(F.softplus(x))

        self.map_reducer = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )

        # 深层回归头
        self.regressor = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

        # --- D. Decoder ---
        self.dec1 = self._up_block(512, 256)
        self.dec2 = self._up_block(256, 128)
        self.dec3 = self._up_block(128, 64)
        self.dec4 = self._up_block(64, 32)
        self.dec5 = self._up_block(32, 16)
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8), nn.ReLU(True),
            nn.Conv2d(8, 1, 3, 1, 1)
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, iq, rough_map):
        iq_feat = self.iq_encoder(iq)
        x1 = self.enc1(rough_map)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)

        x6 = self.map_attention(x6)
        map_flat = x6.view(x6.size(0), -1)
        map_feat = self.map_reducer(map_flat)

        combined = torch.cat([iq_feat, map_feat], dim=1)
        pred_coord = self.regressor(combined)

        d1 = self.dec1(x6)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)
        pred_mask = self.dec6(d5)

        return pred_coord, pred_mask