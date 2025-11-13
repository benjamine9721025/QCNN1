import torch
import torch.nn as nn
import pennylane as qml
import torch.nn.functional as F
from math import log2

# -----------------------------
# Global config (paper-aligned)
# -----------------------------
# Image: 8x8 grayscale as in experiments summary
IMG_H = IMG_W = 8
IN_CHANNELS = 1

# Quantum convolution hyperparams (best per slides: 1 layer, 4 kernels)
KERNEL_SIZE = 4            # 4x4 quantum kernel
STRIDE = 2                 # stride=2
N_KERNELS = 4              # number of quantum kernels
MEAS_AXES = ("X", "Y", "Z") # measure along Bloch X/Y/Z

# Derived sizes
PATCH_SIZE = KERNEL_SIZE * KERNEL_SIZE     # 16
N_POS_QUBITS = int(log2(PATCH_SIZE))       # 4 qubits to amplitude-embed a 4x4 patch

# Each kernel uses a dedicated device/QNode (keeps graphs simple/stable)
# We'll measure on qubit 0 along X/Y/Z, producing 3 channels per kernel.


def _make_qconv_qnode(n_pos_qubits: int):
    """Create a PennyLane QNode that:
    1) Amplitude-encodes a 4x4 patch on 'n_pos_qubits' wires
    2) Applies a simple entangling pattern (ring of CNOTs)
    3) Applies U3(theta, phi, omega) **per qubit** as trainable kernel weights
    4) Measures expvals on PauliX/PauliY/PauliZ for wire 0 => 3 features
    """
    dev = qml.device("default.qubit", wires=n_pos_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(patch_amplitudes, weights):
        # patch_amplitudes: shape (2**n_pos_qubits,), L2-normalized
        # weights: shape (n_pos_qubits, 3) for U3 per qubit
        qml.AmplitudeEmbedding(features=patch_amplitudes, wires=range(n_pos_qubits), normalize=True)

        # simple entangling ring to allow global mixing
        for w in range(n_pos_qubits):
            qml.CNOT(wires=[w, (w + 1) % n_pos_qubits])

        # U3 per qubit = kernel weights (paper: U3 angles are kernel params)
        for w in range(n_pos_qubits):
            th, ph, om = weights[w]
            qml.U3(th, ph, om, wires=w)

        # Measurement along X, Y, Z on qubit 0 -> 3 scalars
        return [
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(0)),
        ]

    return qnode


#class QKernel(nn.Module):
#   """One quantum kernel: produces 3 output channels (X, Y, Z) per patch."""
#    def __init__(self, n_pos_qubits: int):
#        super().__init__()
#        self.qnode = _make_qconv_qnode(n_pos_qubits)
#        # Initialize U3 angles per qubit: (n_pos_qubits, 3)
#        self.weights = nn.Parameter(0.01 * torch.randn(n_pos_qubits, 3))

#    def forward(self, patch_batch: torch.Tensor) -> torch.Tensor:
#        """patch_batch: (B, 2**n_pos_qubits) L2-normalized amplitudes per sample
#        returns: (B, 3)
#        """
#        # Vectorized map over batch using torch.vmap-like loop (explicit loop for clarity)
#        outs = []
#        for p in patch_batch:
#            outs.append(self.qnode(p, self.weights))
#        return torch.stack(outs, dim=0)  # (B, 3)

class QKernel(nn.Module):
    """One quantum kernel: produces 3 output channels (X, Y, Z) per patch."""
    def __init__(self, n_pos_qubits: int):
        super().__init__()
        self.qnode = _make_qconv_qnode(n_pos_qubits)
        # Initialize U3 angles per qubit: (n_pos_qubits, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(n_pos_qubits, 3))

    def forward(self, patch_batch: torch.Tensor) -> torch.Tensor:
        """patch_batch: (B, 2**n_pos_qubits) L2-normalized amplitudes per sample
        returns: (B, 3)
        """
        outs = []
        for p in patch_batch:                 # p: (16,)
            q_out = self.qnode(p, self.weights)
            # q_out 通常是長度 3 的 list/tuple，每個元素是 scalar tensor
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(q_out)    # 變成 shape: (3,)
            outs.append(q_out)                # list 裡面都是 tensor(3,)

        return torch.stack(outs, dim=0)       # (B, 3)





class QConv2d(nn.Module):
    """Quantum convolution layer using amplitude encoding + U3 kernels.
    Sliding 4x4 patches with stride=2 over an 8x8 image.
    Produces N_KERNELS * 3 channels (per-kernel X/Y/Z) with spatial size 3x3.
    """
    def __init__(self, kernel_size=KERNEL_SIZE, stride=STRIDE, n_kernels=N_KERNELS):
        super().__init__()
        assert kernel_size == 4, "This implementation assumes 4x4 patches (can be generalized)."
        assert stride == 2, "This implementation assumes stride=2 (can be generalized)."
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_kernels = n_kernels

        # Build quantum kernels
        self.qkernels = nn.ModuleList([QKernel(N_POS_QUBITS) for _ in range(n_kernels)])

    @staticmethod
    def _extract_patches(x: torch.Tensor, k: int, s: int) -> torch.Tensor:
        """x: (B, 1, H, W) -> patches: (B, L, k*k)
        where L = number of sliding windows = ((H-k)/s + 1) * ((W-k)/s + 1)
        """
        B, C, H, W = x.shape
        assert C == 1, "grayscale only"
        patches = []
        for i in range(0, H - k + 1, s):
            for j in range(0, W - k + 1, s):
                patch = x[:, :, i:i+k, j:j+k].reshape(B, -1)  # (B, k*k)
                patches.append(patch)
        return torch.stack(patches, dim=1)  # (B, L, k*k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 8, 8) in [0,1] or arbitrary real range
        B, C, H, W = x.shape
        patches = self._extract_patches(x, self.kernel_size, self.stride)  # (B, L, 16)
        B, L, D = patches.shape

        # L2 normalize to form amplitudes per patch (AmplitudeEmbedding requires norm==1)
        eps = 1e-12
        norms = torch.linalg.vector_norm(patches, dim=-1, keepdims=True) + eps
        amps = patches / norms  # (B, L, 16)

        # For each kernel, run QNode on all L patches (batched)
        kernel_outputs = []
        for qk in self.qkernels:
            # collapse (B*L, 16)
            amps_bl = amps.reshape(B*L, D)
            out_bl = qk(amps_bl)  # (B*L, 3)
            out = out_bl.reshape(B, L, 3)  # (B, L, 3)
            kernel_outputs.append(out)

        # concat over kernels -> (B, L, 3*N_KERNELS)
        feats = torch.cat(kernel_outputs, dim=-1)

        # reshape L back to spatial grid: H_out = W_out = 3 for 8x8 with k=4, s=2
        H_out = W_out = int((H - self.kernel_size) / self.stride) + 1
        feats = feats.permute(0, 2, 1).reshape(B, 3*self.n_kernels, H_out, W_out)
        return feats  # (B, 12, 3, 3)


class QCCNN(nn.Module):
    """Hybrid QC-CNN: quantum feature extractor + classical head.
    Output classes is configurable (default 10).
    """
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.qconv = QConv2d(kernel_size=KERNEL_SIZE, stride=STRIDE, n_kernels=N_KERNELS)
        self.act = nn.LeakyReLU(0.1)
        # 12 channels * 3 * 3 = 108 features
        self.fc1 = nn.Linear(12*3*3, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        接受：
        - (B, 1, 8, 8) 影像張量
        - (B, 64)      已攤平成 8x8 的向量
        - (B, N)       N >= 64，例如 mnist_179_1200.csv 這種「多特徵向量」
                        在這種情況下，我們取前 64 維當作 8x8 灰階影像。
        """
        B = x.shape[0]

        if x.dim() == 2:
            # x: (B, N) → 至少要有 64 維才能 reshape 成 8x8
            if x.shape[1] < IMG_H * IMG_W:
                raise ValueError(
                    f"Expected at least {IMG_H*IMG_W} features to reshape into 8x8, got {x.shape[1]}"
                )
            # 只取前 64 維來當 8x8 影像
            x = x[:, :IMG_H * IMG_W]
            x = x.view(B, 1, IMG_H, IMG_W)
        elif x.dim() == 4 and x.shape[1:] == (1, IMG_H, IMG_W):
            # 已經是 (B,1,8,8)，不用處理
            pass
        else:
            raise AssertionError(
                f"Expected input of shape (B,1,{IMG_H},{IMG_W}) or flat (B,>= {IMG_H*IMG_W}) "
                f"features, got {tuple(x.shape)}"
            )

        x = self.qconv(x)                 # (B, 12, 3, 3)
        x = self.act(x)
        x = x.reshape(B, -1) 
        
    # ⭐ 關鍵：把 x 轉成 fc1 權重的 dtype（通常是 float32）
        x = x.to(dtype=self.fc1.weight.dtype)
        
        x = self.act(self.fc1(x))         # (B, 32)
        x = self.fc2(x)                   # (B, n_classes)
        return x
    

# -------------------
# Quick sanity check
# -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    model = QCCNN(n_classes=10)
    dummy = torch.randn(2, 1, 8, 8)
    out = model(dummy)
    print("Output shape:", out.shape)  # (2, 10)




