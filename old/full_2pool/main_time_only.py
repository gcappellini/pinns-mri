import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------
# Simple MLP: input t_scaled -> 6 outputs
# -------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=6, width=64, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------------
# PINN for 2-pool Bloch–McConnell with RF
# time-only input
# -------------------------------
class BlochMcConnellPINNTimeOnly(nn.Module):
    def __init__(self, T1w, T2w, T1s, T2s,
                 k_ws, k_sw, delta_omega_w, delta_omega_s,
                 omega1, M_w0, M_s0,
                 t_max):
        super().__init__()

        # Physical parameters (constants for this experiment)
        self.T1w = T1w
        self.T2w = T2w
        self.T1s = T1s
        self.T2s = T2s
        self.k_ws = k_ws              # water -> solute
        self.k_sw = k_sw              # solute -> water
        self.delta_omega_w = delta_omega_w
        self.delta_omega_s = delta_omega_s
        self.omega1 = omega1
        self.M_w0 = M_w0
        self.M_s0 = M_s0

        # Time scaling: τ = t / t_max
        self.t_max = t_max
        self.time_scale = t_max       # dM/dτ = t_max * dM/dt

        # Neural network
        self.network = MLP(in_dim=1, out_dim=6, width=64, depth=4)

    def forward(self, t):
        """
        t: (batch,) physical time in seconds
        returns 6 components: (M_wx, M_wy, M_wz, M_sx, M_sy, M_sz)
        """
        t_scaled = t / self.t_max
        x = t_scaled.unsqueeze(1)      # (batch,1)
        out = self.network(x)
        M_wx, M_wy, M_wz, M_sx, M_sy, M_sz = torch.chunk(out, 6, dim=-1)
        return M_wx.squeeze(-1), M_wy.squeeze(-1), M_wz.squeeze(-1), \
               M_sx.squeeze(-1), M_sy.squeeze(-1), M_sz.squeeze(-1)

    def compute_physics_loss(self, t_collocation):
        """
        Physics residual of all 6 Bloch–McConnell equations at collocation times
        """
        t_collocation.requires_grad_(True)

        # Forward pass
        M_wx, M_wy, M_wz, M_sx, M_sy, M_sz = self.forward(t_collocation)

        # Time derivatives dM/dτ via autograd (τ = t / t_max)
        dMwx_dt_scaled = torch.autograd.grad(M_wx, t_collocation,
                                             grad_outputs=torch.ones_like(M_wx),
                                             create_graph=True)[0]
        dMwy_dt_scaled = torch.autograd.grad(M_wy, t_collocation,
                                             grad_outputs=torch.ones_like(M_wy),
                                             create_graph=True)[0]
        dMwz_dt_scaled = torch.autograd.grad(M_wz, t_collocation,
                                             grad_outputs=torch.ones_like(M_wz),
                                             create_graph=True)[0]
        dMsx_dt_scaled = torch.autograd.grad(M_sx, t_collocation,
                                             grad_outputs=torch.ones_like(M_sx),
                                             create_graph=True)[0]
        dMsy_dt_scaled = torch.autograd.grad(M_sy, t_collocation,
                                             grad_outputs=torch.ones_like(M_sy),
                                             create_graph=True)[0]
        dMsz_dt_scaled = torch.autograd.grad(M_sz, t_collocation,
                                             grad_outputs=torch.ones_like(M_sz),
                                             create_graph=True)[0]

        # dM/dτ = t_max * dM/dt
        time_scale = self.time_scale

        # Unpack constants for readability
        k_ws = self.k_ws
        k_sw = self.k_sw
        dw_w = self.delta_omega_w
        dw_s = self.delta_omega_s
        omega1 = self.omega1

        # Water pool (a)
        residual_wx = dMwx_dt_scaled * time_scale - (
            -k_ws * M_wx + dw_w * M_wy +
            k_sw * M_sx - M_wx / self.T2w
        )

        residual_wy = dMwy_dt_scaled * time_scale - (
            dw_w * M_wx - k_ws * M_wy -
            omega1 * M_wz - M_wy / self.T2w
        )

        residual_wz = dMwz_dt_scaled * time_scale - (
            omega1 * M_wy - k_ws * M_wz +
            k_sw * M_sz + (self.M_w0 - M_wz) / self.T1w
        )

        # Solute pool (b)
        residual_sx = dMsx_dt_scaled * time_scale - (
            -k_sw * M_sx + dw_s * M_sy +
            k_ws * M_wx - M_sx / self.T2s
        )

        residual_sy = dMsy_dt_scaled * time_scale - (
            dw_s * M_sx - k_sw * M_sy -
            omega1 * M_sz - M_sy / self.T2s
        )

        residual_sz = dMsz_dt_scaled * time_scale - (
            omega1 * M_sy - k_sw * M_sz +
            k_ws * M_wz + (self.M_s0 - M_sz) / self.T1s
        )

        physics_loss = (
            residual_wx.pow(2) + residual_wy.pow(2) + residual_wz.pow(2) +
            residual_sx.pow(2) + residual_sy.pow(2) + residual_sz.pow(2)
        ).mean()

        return physics_loss

    def compute_ic_loss(self, device):
        """
        IC at t=0: M_wx=My=0, M_wz=M_w0; M_sx=My=0, M_sz=M_s0
        """
        t_zero = torch.zeros(1, device=device)
        M_wx, M_wy, M_wz, M_sx, M_sy, M_sz = self.forward(t_zero)

        ic_loss = (
            M_wx.pow(2) +
            M_wy.pow(2) +
            (M_wz - self.M_w0).pow(2) +
            M_sx.pow(2) +
            M_sy.pow(2) +
            (M_sz - self.M_s0).pow(2)
        ).mean()

        return ic_loss


# -------------------------------
# Minimal training loop (example)
# -------------------------------
def train_pinn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example parameters (replace with your actual values)
    T1w = 1.5   # s
    T2w = 0.1
    T1s = 1.0
    T2s = 0.05
    k_ws = 5.0  # Hz or s^-1, just be consistent with MATLAB
    k_sw = 5.0
    delta_omega_w = 0.0
    delta_omega_s = 100.0
    omega1 = 50.0
    M_w0 = 1.0
    M_s0 = 0.0
    t_max = 5.0  # s, total simulation time

    model = BlochMcConnellPINNTimeOnly(
        T1w, T2w, T1s, T2s,
        k_ws, k_sw, delta_omega_w, delta_omega_s,
        omega1, M_w0, M_s0,
        t_max
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5000):
        model.train()
        optimizer.zero_grad()

        # Sample collocation times in [0, t_max]
        t_collocation = torch.rand(1024, device=device) * t_max

        physics_loss = model.compute_physics_loss(t_collocation)
        ic_loss = model.compute_ic_loss(device)

        loss = physics_loss + ic_loss

        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.3e}, "
                  f"Phys: {physics_loss.item():.3e}, IC: {ic_loss.item():.3e}")

    # After training, you can sample model on a grid of t
    # and compare to MATLAB ground truth in a separate script.
    return model


if __name__ == "__main__":
    trained_model = train_pinn()
