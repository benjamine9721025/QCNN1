import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp


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
        """
        amps: 來自 PyTorch 的一維張量，長度 = 2**n_pos_qubits
        在這裡做「最後一關」的清洗與正規化，避免 NaN / 0-norm
        """
        # 轉成 PennyLane 的 numpy 陣列（視為純 data，不需要對 amps 求梯度）
        a = pnp.array(amps, dtype=float)

        # 1) 先把 NaN / Inf 統一成 0
        a = pnp.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) 計算平方範數
        sqnorm = pnp.sum(pnp.abs(a) ** 2)
        norm = pnp.sqrt(sqnorm)

        # 3) 如果 norm 太小或是 NaN，就手動改成 |1000...0> 基底態
        if not pnp.isfinite(norm) or norm < 1e-8:
            a = pnp.zeros_like(a)
            a[0] = 1.0   # → norm = 1
        else:
            # 否則就正常正規化
            a = a / norm

        # 4) 這裡我們就不用再讓 AmplitudeEmbedding normalize 了
        qml.AmplitudeEmbedding(a, wires=range(n_pos_qubits), normalize=False)

        # 簡單的一層參數化旋轉 + entangling 結構
        for w in range(n_pos_qubits):
            qml.Rot(weights[w, 0], weights[w, 1], weights[w, 2], wires=w)

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
        # 先保險一次：把 NaN / Inf 打掉
        patch_batch = torch.nan_to_num(patch_batch, nan=0.0, posinf=0.0, neginf=0.0)

        # 再次保證每一個向量的 norm=1
        norms = torch.linalg.vector_norm(patch_batch, dim=-1, keepdims=True)  # (B,1)
        bad_mask = (norms < 1e-8) | torch.isnan(norms)

        safe_norms = torch.where(bad_mask, torch.ones_like(norms), norms)
        patch_batch = patch_batch / safe_norms

        if bad_mask.any():
            bad_idx = bad_mask.squeeze(-1)
            patch_batch[bad_idx] = 0.0
            patch_batch[bad_idx, 0] = 1.0

        outs = []
        for p in patch_batch:  # p: (2**n_pos_qubits,)
            q_out = self.qnode(p, self.weights)

            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(q_out)
            else:
                q_out = torch.as_tensor(q_out)

            q_out = torch.nan_to_num(q_out, nan=0.0, posinf=1.0, neginf=-1.0)
            outs.append(q_out)

        return torch.stack(outs, dim=0)




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

        # 取得所有 4x4 patches
        patches = (
            x.unfold(2, self.kernel_size, self.stride)
             .unfold(3, self.kernel_size, self.stride)
        )  # (B, 1, H_out, W_out, k, k)

        B, C, H_out, W_out, k, k2 = patches.shape
        assert k == self.kernel_size and k2 == self.kernel_size

        # 攤平成 (B * H_out * W_out, k*k)
        patches = patches.contiguous().view(B * H_out * W_out, k * k)  # (N, 16)

        # ---------- 第一次：處理 0 patch ----------
        # 計算每個 patch 的 L2 norm
        norms = torch.linalg.vector_norm(patches, dim=-1, keepdims=True)  # (N, 1)

        # 找出完全為 0 的 patch
        zero_mask = norms < 1e-8  # (N,1) bool

        # 避免除以 0：對於 zero patch，先暫時把 norm 設成 1
        safe_norms = torch.where(zero_mask, torch.ones_like(norms), norms)

        # 正規化
        amps = patches / safe_norms  # (N, k*k)

        # 把完全為 0 的 patch，手動指定為 |1000...0>（合法且 norm=1）
        if zero_mask.any():
            zero_idx = zero_mask.squeeze(-1)  # (N,)
            amps[zero_idx] = 0.0
            amps[zero_idx, 0] = 1.0

        # ---------- 第二次：清掉 NaN / Inf ----------
        amps = torch.nan_to_num(amps, nan=0.0, posinf=0.0, neginf=0.0)

        # 再保險一次：重新 normalize，處理前面 nan_to_num 可能造成的微小偏差
        norms2 = torch.linalg.vector_norm(amps, dim=-1, keepdims=True)  # (N,1)
        bad_mask = (norms2 < 1e-8) | torch.isnan(norms2)

        safe_norms2 = torch.where(bad_mask, torch.ones_like(norms2), norms2)
        amps = amps / safe_norms2

        if bad_mask.any():
            bad_idx = bad_mask.squeeze(-1)
            amps[bad_idx] = 0.0
            amps[bad_idx, 0] = 1.0


        
        # 最後做一次 NaN 防護
        amps = torch.nan_to_num(amps, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- 送進每個量子 kernel ----------
        kernel_outputs = []
        for qk in self.qkernels:
            out_bl = qk(amps)  # (N, 3)
            out = out_bl.view(B, H_out, W_out, 3)  # (B, H_out, W_out, 3)
            kernel_outputs.append(out)

        feats = torch.cat(kernel_outputs, dim=-1)   # (B, H_out, W_out, 3*n_kernels)
        feats = feats.permute(0, 3, 1, 2).contiguous()  # (B, 3*n_kernels, H_out, W_out)
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


























