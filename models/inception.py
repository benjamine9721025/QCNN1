import torch
import torch.nn as nn
import pennylane as qml

# =============================================================
# 全域超參數
# =============================================================
IMG_H = 8
IMG_W = 8


KERNEL_SIZE = 4
STRIDE = 2
N_KERNELS = 4           # 量子卷積 kernel 數量
N_POS_QUBITS = 4        # 4x4 patch → 16 = 2**4 → 4 qubits


# ============================================================
# 建立量子卷積用的 QNode
#   - input: amplitudes (len = 2**n_pos_qubits)
#   - weights: (n_pos_qubits, 3)
#   - output: (3,) = <X>, <Y>, <Z> on qubit 
# ============================================================
def _make_qconv_qnode(n_pos_qubits: int):
    dev = qml.device("default.qubit", wires=n_pos_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(amps, weights):
        # amps: shape (2**n_pos_qubits,)
        # 在外部我們已經做 L2 normalize，這裡 normalize=False
        qml.AmplitudeEmbedding(amps, wires=range(n_pos_qubits), normalize=False)

        # 簡單的一層參數化旋轉 + entangling 結構
        for w in range(n_pos_qubits):
            qml.Rot(weights[w, 0], weights[w, 1], weights[w, 2], wires=w)

        # 這裡可以加一些 entangling，如果想要更複雜：
        for w in range(n_pos_qubits - 1):
            qml.CNOT(wires=[w, w + 1])
        qml.CNOT(wires=[n_pos_qubits - 1, 0])

        # 只量測 qubit 0 的 X/Y/Z 期望值
        return (
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(0)),
        )

    return circuit


# ============================================================
# 單一量子 kernel：對一個 patch 產生 3 維輸出
# ============================================================
class QKernel(nn.Module):
    """One quantum kernel: produces 3 output channels (X, Y, Z) per patch."""

    def __init__(self, n_pos_qubits: int):
        super().__init__()
        self.qnode = _make_qconv_qnode(n_pos_qubits)
        # trainable parameters: (n_pos_qubits, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(n_pos_qubits, 3))

    def forward(self, patch_batch: torch.Tensor) -> torch.Tensor:
        """
        patch_batch: (B, 2**n_pos_qubits) L2-normalized amplitudes per sample
        returns: (B, 3)
        """
        outs = []
        for p in patch_batch:  # p: (2**n_pos_qubits,)
            q_out = self.qnode(p, self.weights)

            # qnode 通常回傳 list/tuple[scalar tensor]，先轉成 1D tensor
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(q_out)
            else:
                q_out = torch.as_tensor(q_out)

            # 🔒 NaN/Inf 防護
            q_out = torch.nan_to_num(q_out, nan=0.0, posinf=1.0, neginf=-1.0)

            outs.append(q_out)  # (3,)

        return torch.stack(outs, dim=0)  # (B, 3)


# ============================================================
# 量子「卷積層」：每個 4x4 patch → amplitude encoding → QKernel
# ============================================================
class QConv2d(nn.Module):
    """
    輸入: x (B, 1, H, W)
    流程:
      - 以 kernel_size/stride 擷取 4x4 patches
      - 每個 patch flatten 成長度 16 的向量
      - L2 normalize → 當作 amplitude encoding
      - 每個量子 kernel QKernel 產出 3 維特徵 (X, Y, Z)
    輸出: (B, 3 * n_kernels, H_out, W_out)
    """

    def __init__(self, kernel_size: int, stride: int, n_kernels: int, n_pos_qubits: int = N_POS_QUBITS):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_kernels = n_kernels
        self.n_pos_qubits = n_pos_qubits

        # 建立多個量子 kernel
        self.qkernels = nn.ModuleList(
            [QKernel(n_pos_qubits=self.n_pos_qubits) for _ in range(self.n_kernels)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W)
        return: (B, 3 * n_kernels, H_out, W_out)
        """
        assert x.dim() == 4, f"QConv2d expects 4D input (B,1,H,W), got {x.shape}"
        B, C, H, W = x.shape
        assert C == 1, f"QConv2d expects single channel input, got C={C}"

        # 使用 unfold 取得所有 4x4 patches
        patches = (
            x.unfold(2, self.kernel_size, self.stride)   # dim=2 → H
             .unfold(3, self.kernel_size, self.stride)   # dim=3 → W
        )  # (B, 1, H_out, W_out, k, k)

        B, C, H_out, W_out, k, k2 = patches.shape
        assert k == self.kernel_size and k2 == self.kernel_size

        # 攤平成 (B * H_out * W_out, k*k)
        patches = patches.contiguous().view(B * H_out * W_out, k * k)

        # L2 normalize → amplitude vector
        eps = 1e-12
        norms = torch.linalg.vector_norm(patches, dim=-1, keepdims=True) + eps
        amps = patches / norms  # (B * H_out * W_out, k*k)
        amps = torch.nan_to_num(amps, nan=0.0, posinf=0.0, neginf=0.0)

        # 通過每個量子 kernel
        kernel_outputs = []
        for qk in self.qkernels:
            out_bl = qk(amps)  # (B*H_out*W_out, 3)
            out = out_bl.view(B, H_out, W_out, 3)  # (B, H_out, W_out, 3)
            kernel_outputs.append(out)

        # 在最後一個維度串接 kernels 的輸出 → (B, H_out, W_out, 3 * n_kernels)
        feats = torch.cat(kernel_outputs, dim=-1)

        # 變換維度為 (B, 3*n_kernels, H_out, W_out)
        feats = feats.permute(0, 3, 1, 2).contiguous()

        # 再做一次 NaN 防護
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats


# ============================================================
# 整體 QC-CNN 分類模型
#   - 接受 (B, 1, 8, 8) 或 (B, N>=64) flatten 特徵
#   - 先擷取前 64 維 reshape 成 8x8 灰階
#   - 經過量子卷積 QConv2d
#   - flatten → fc1(108→32) → fc2(32→n_classes)
#   - 輸入一張 8×8 灰階圖（或至少 64 維的特徵），先用量子卷積層抽特徵，再接兩層全連接做分類。
# ============================================================
class QCCNN(nn.Module):
   def __init__(self, n_classes=3):
        super().__init__()

        self.feat_proj = nn.Linear(196, 64)

        
        self.qconv = QConv2d(kernel_size=KERNEL_SIZE, stride=STRIDE, n_kernels=N_KERNELS)
        self.act = nn.LeakyReLU(0.1)

        # QConv 輸出 shape: (B, 3*N_KERNELS, H_out, W_out)
        # H_out = W_out = (8 - 4)/2 + 1 = 3
        conv_out_dim = 3 * N_KERNELS * 3 * 3  # = 12 * 3 * 3 = 108

        self.fc1 = nn.Linear(conv_out_dim, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def _ensure_image(self, x):
        """
        輸入可以是：
        - (B, 196) 數值特徵 → 線性降維 → 64 → reshape 成 (B,1,8,8)
        - (B, 64) 已經是 64 維 → 直接 reshape 成 (B,1,8,8)
        - (B,1,H,W) 原本就是圖片 → 視情況 center-crop 等
        """
        B = x.shape[0]

       # Case 1：扁平特徵 (B, N)
        if x.dim() == 2:
            feat_dim = x.shape[1]

            if feat_dim == 196:
                # ⭐ 196 維 → Linear 降維到 64 維
                x = self.feat_proj(x)   # (B, 64)
            elif feat_dim >= IMG_H * IMG_W:
                # 如果不是 196，但至少有 64 維，就直接取前 64 維
                x = x[:, : IMG_H * IMG_W]  # (B, 64)
            else:
                raise ValueError(
                    f"Expected at least {IMG_H*IMG_W} features to reshape into 8x8, "
                    f"got {feat_dim}"
                )

            x = x.view(B, 1, IMG_H, IMG_W)   # (B,1,8,8)
            x = x / 255.0
            return x

        # Case 2：原本就是圖片 (B,1,8,8)
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == IMG_H and x.shape[3] == IMG_W:
            x = x / 255.0
            return x

        # 其他情況都視為錯誤
        raise AssertionError(
            f"Expected input of shape (B,1,{IMG_H},{IMG_W}) "
            f"or flat (B,>= {IMG_H*IMG_W}), got {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 轉成 (B,1,8,8)
        x = self._ensure_image(x)

        # 量子卷積
        x = self.qconv(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # flatten + 全連接
        x = self.act(x)
        x = x.reshape(B, -1)                 # (B,108)
        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self._ensure_image(x)           # (B,1,8,8) + /255

        x = self.qconv(x)                   # 這裡允許反向傳遞
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.act(x)
        x = x.reshape(B, -1)                # (B,108)
        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x




















