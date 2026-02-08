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
import pennylane as qml
import numpy as np

def _make_qconv_qnode(n_pos_qubits: int):
    """建立一個『角度編碼』的量子卷積核 qnode（不走 torch interface）"""

    dev = qml.device("default.qubit", wires=n_pos_qubits)

    # ❗ 不要再寫 interface="torch"
    @qml.qnode(dev)
    def circuit(features, weights):
        """
        features: numpy array, shape = (n_pos_qubits,)
        weights:  numpy array, shape = (n_pos_qubits, 3)
        """

        # 確保是 numpy，且沒有 NaN / Inf
        f = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        w = np.nan_to_num(np.asarray(weights,  dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        # 角度編碼：每個 qubit 一個 RY(f[w])
        for i in range(n_pos_qubits):
            angle = f[i] if i < len(f) else 0.0
            qml.RY(angle, wires=i)

        # 參數化旋轉層
        for i in range(n_pos_qubits):
            qml.Rot(w[i, 0], w[i, 1], w[i, 2], wires=i)

        # entangling
        for i in range(n_pos_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_pos_qubits - 1, 0])

        # 測量 qubit 0 的 X/Y/Z 期望值
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
    """一個量子卷積核：輸入 4 維 feature，輸出 3 維 (X,Y,Z)"""

    def __init__(self, n_pos_qubits: int = N_POS_QUBITS):
        super().__init__()
        self.n_pos_qubits = n_pos_qubits
        self.qnode = _make_qconv_qnode(n_pos_qubits)

        # 🔸 注意：這裡依然是 torch.nn.Parameter
        # 但是我們在 forward 時會 .detach()，所以不會對它做反傳
        self.weights = nn.Parameter(
            0.01 * torch.randn(self.n_pos_qubits, 3, dtype=torch.float32)
        )

    def forward(self, feature_batch: torch.Tensor) -> torch.Tensor:
        """
        feature_batch: (B, n_pos_qubits) 的 torch.Tensor
        return:       (B, 3) 的 torch.Tensor
        """

        # 先清掉 NaN / Inf
        feature_batch = torch.nan_to_num(feature_batch, nan=0.0, posinf=0.0, neginf=0.0)

        # 👉 關鍵：量子電路用的是「純 numpy」資料，不走 autograd
        fb_np = feature_batch.detach().cpu().numpy()
        w_np  = self.weights.detach().cpu().numpy()

        outs = []
        for f in fb_np:  # f: (n_pos_qubits,)
            q_out = self.qnode(f, w_np)  # qnode 回傳的是 numpy array / tuple

            # 轉回 torch.Tensor，但不需要 gradient
            q_out = torch.tensor(q_out, dtype=torch.float32)
            outs.append(q_out)

        # 最後把所有 sample 堆成 (B, 3)
        return torch.stack(outs, dim=0)






# ============================================================
# 量子「卷積層」：每個 4x4 patch → amplitude encoding → QKernel
# ============================================================
class QConv2d(nn.Module):
    """
    輸入: x (B, 1, H, W)
    輸出: (B, 3 * n_kernels, H_out, W_out)
    """

    def __init__(self, kernel_size: int, stride: int, n_kernels: int, n_pos_qubits: int = N_POS_QUBITS):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_kernels = n_kernels
        self.n_pos_qubits = n_pos_qubits

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

        # 取得所有 4x4 patches: (B, 1, H_out, W_out, k, k)
        patches = (
            x.unfold(2, self.kernel_size, self.stride)
             .unfold(3, self.kernel_size, self.stride)
        )

        B, C, H_out, W_out, k, k2 = patches.shape
        assert k == self.kernel_size and k2 == self.kernel_size

        # 現在我們不用 amplitude，而是做簡單的「列平均」→ 4 維角度 feature:
        # 每一個 patch 形狀是 (k, k)，我們取每一列平均 → (k,)
        # 對於 4x4 patch，就是 4 維向量。
        patches = patches.view(B, H_out, W_out, k, k)  # 明確 reshape

        # (B, H_out, W_out, k, k) → (B * H_out * W_out, k, k)
        patches_flat = patches.contiguous().view(B * H_out * W_out, k, k)

        # 對「列」做平均，得到 (N, k) = (N, 4)
        features = patches_flat.mean(dim=-1)  # 在最後一維 (列) 上平均 → (N, k)
        # 再做一次保護
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 送進每一個量子 kernel
        kernel_outputs = []
        for qk in self.qkernels:
            out_bl = qk(features)  # (N, 3)
            out = out_bl.view(B, H_out, W_out, 3)  # (B, H_out, W_out, 3)
            kernel_outputs.append(out)

        # 最後組成 (B, 3 * n_kernels, H_out, W_out)
        feats = torch.cat(kernel_outputs, dim=-1)            # (B, H_out, W_out, 3*n_kernels)
        feats = feats.permute(0, 3, 1, 2).contiguous()       # (B, 3*n_kernels, H_out, W_out)
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

        # 196 → 64
        self.feat_proj = nn.Linear(196, 64)

        self.qconv = QConv2d(
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            n_kernels=N_KERNELS
        )
        self.act = nn.LeakyReLU(0.1)

        # QConv 輸出: (B, 3*N_KERNELS, 3, 3) = 108
        conv_out_dim = 3 * N_KERNELS * 3 * 3

        self.fc1 = nn.Linear(conv_out_dim, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def _ensure_image(self, x):
        """
        x:
        - (B,196) → Linear → (B,64) → reshape (B,1,8,8)
        - (B,64)  → reshape (B,1,8,8)
        - (B,1,8,8) → 直接使用
        """
        B = x.shape[0]

        if x.dim() == 2:
            feat_dim = x.shape[1]

            if feat_dim == 196:
                x = self.feat_proj(x)
            elif feat_dim >= IMG_H * IMG_W:
                x = x[:, : IMG_H * IMG_W]
            else:
                raise ValueError(
                    f"Need >= {IMG_H*IMG_W} features, got {feat_dim}"
                )

            x = x.view(B, 1, IMG_H, IMG_W)
            return x / 255.0

        if x.dim() == 4 and x.shape[1:] == (1, IMG_H, IMG_W):
            return x / 255.0

        raise AssertionError(f"Invalid input shape: {x.shape}")

    def forward(self, x):
        B = x.shape[0]

        x = self._ensure_image(x)
        x = self.qconv(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.act(x)
        x = x.reshape(B, -1)
        x = x.to(self.fc1.weight.dtype)

        x = self.act(self.fc1(x))
        return self.fc2(x)






























