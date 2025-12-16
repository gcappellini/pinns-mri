import torch
import torch.nn as nn
import numpy as np

class FourierFeatures(nn.Module):
    """Fourier feature mapping for improved neural network training on periodic functions."""
    
    def __init__(self, input_dim, num_frequencies=10, scale=1.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Random Fourier features - sample from Gaussian
        self.B = nn.Parameter(torch.randn(input_dim, num_frequencies) * scale, requires_grad=False)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, input_dim)
        Returns:
            Fourier features of shape (batch, 2*num_frequencies*input_dim)
        """
        x_proj = 2 * np.pi * x @ self.B  # (batch, num_frequencies)
        result = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return result


class InputScaler:
    """Scale and normalize inputs to appropriate ranges."""
    
    def __init__(self, t_max=1.0, k_range=(10, 10000), delta_omega_max=1000, 
                 omega1_max=500, M0_range=(0.001, 1.0)):
        """
        Args:
            t_max: Maximum saturation time (seconds)
            k_range: (min, max) exchange rates in Hz
            delta_omega_max: Maximum offset frequency in Hz (or rad/s)
            omega1_max: Maximum RF field strength in Hz (or rad/s)
            M0_range: (min, max) equilibrium magnetization range
        """
        self.t_max = t_max
        self.k_min, self.k_max = np.log10(k_range[0]), np.log10(k_range[1])
        self.delta_omega_max = delta_omega_max
        self.omega1_max = omega1_max
        self.M0_min, self.M0_max = M0_range
    
    def scale(self, t, k_ws, k_sw, delta_omega_s, delta_omega_w, omega1, M_w0, M_s0):
        """
        Scale all inputs to approximately [-1, 1] or [0, 1] range.
        
        Args:
            All inputs can be tensors of shape (batch,) or scalars
        Returns:
            Tuple of scaled inputs, all with shape (batch,) if input was batched
        """
        # Time: normalize to [0, 1]
        t_scaled = t / self.t_max
        
        # Exchange rates: log scale then normalize to [-1, 1]
        k_ws_log = torch.log10(k_ws)
        k_sw_log = torch.log10(k_sw)
        k_ws_scaled = 2 * (k_ws_log - self.k_min) / (self.k_max - self.k_min) - 1
        k_sw_scaled = 2 * (k_sw_log - self.k_min) / (self.k_max - self.k_min) - 1
        
        # Offset frequencies: normalize to [-1, 1]
        delta_omega_s_scaled = delta_omega_s / self.delta_omega_max
        delta_omega_w_scaled = delta_omega_w / self.delta_omega_max
        
        # RF field: normalize to [0, 1]
        omega1_scaled = omega1 / self.omega1_max
        
        # Equilibrium magnetizations: normalize to [0, 1]
        M_w0_scaled = (M_w0 - self.M0_min) / (self.M0_max - self.M0_min)
        M_s0_scaled = (M_s0 - self.M0_min) / (self.M0_max - self.M0_min)
        
        return (t_scaled, k_ws_scaled, k_sw_scaled, delta_omega_s_scaled, 
                delta_omega_w_scaled, omega1_scaled, M_w0_scaled, M_s0_scaled)
    
    def unscale_k(self, k_scaled):
        """Convert scaled exchange rate back to Hz."""
        k_log = (k_scaled + 1) / 2 * (self.k_max - self.k_min) + self.k_min
        return 10 ** k_log


class BlochMcConnellPINN(nn.Module):
    """
    Physics-Informed Neural Network for 2-pool Bloch-McConnell equations.
    """
    
    def __init__(self, hidden_dims=[128, 128, 128, 128], 
                 num_fourier_features=10, fourier_scale=1.0,
                 T1w=1.5, T2w=0.05, T1s=1.0, T2s=0.01):
        """
        Args:
            hidden_dims: List of hidden layer dimensions
            num_fourier_features: Number of frequencies for Fourier features
            fourier_scale: Scale parameter for Fourier feature sampling
            T1w, T2w: Water pool relaxation times (seconds)
            T1s, T2s: Solute pool relaxation times (seconds)
        """
        super().__init__()
        
        # Fixed relaxation parameters
        self.T1w = T1w
        self.T2w = T2w
        self.T1s = T1s
        self.T2s = T2s
        
        # Fourier feature mapping
        self.fourier = FourierFeatures(input_dim=8, 
                                       num_frequencies=num_fourier_features,
                                       scale=fourier_scale)
        
        # Input dimension after Fourier features
        fourier_dim = 2 * num_fourier_features #* 8
        
        # Build MLP
        layers = []
        in_dim = fourier_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        # Output layer: 6 magnetization components
        layers.append(nn.Linear(in_dim, 6))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, t, k_ws, k_sw, delta_omega_s, delta_omega_w, omega1, M_w0, M_s0):
        """
        Forward pass through the network.
        
        Args:
            All inputs should be scaled and have shape (batch,)
        Returns:
            Magnetization components: (M_wx, M_wy, M_wz, M_sx, M_sy, M_sz)
            Each with shape (batch,)
        """
        # Stack inputs
        x = torch.stack([t, k_ws, k_sw, delta_omega_s, delta_omega_w, 
                        omega1, M_w0, M_s0], dim=1)  # (batch, 8)
        
        # Apply Fourier features
        x_fourier = self.fourier(x)  # (batch, fourier_dim)
        
        # Pass through network
        output = self.network(x_fourier)  # (batch, 6)
        
        # Split into components
        M_wx = output[:, 0]
        M_wy = output[:, 1]
        M_wz = output[:, 2]
        M_sx = output[:, 3]
        M_sy = output[:, 4]
        M_sz = output[:, 5]
        
        return M_wx, M_wy, M_wz, M_sx, M_sy, M_sz
    
    def compute_physics_loss(self, t, k_ws, k_sw, delta_omega_s, delta_omega_w, 
                            omega1, M_w0, M_s0, scaler):
        """
        Compute the physics-informed loss (ODE residuals).
        
        Args:
            All inputs are SCALED
            scaler: InputScaler object to unscale exchange rates
        Returns:
            physics_loss: Mean squared residual of the 6 ODEs
        """
        # Enable gradient computation for time
        t.requires_grad_(True)
        
        # Forward pass
        M_wx, M_wy, M_wz, M_sx, M_sy, M_sz = self.forward(
            t, k_ws, k_sw, delta_omega_s, delta_omega_w, omega1, M_w0, M_s0
        )
        
        # Compute time derivatives using autograd
        dMwx_dt = torch.autograd.grad(M_wx, t, grad_outputs=torch.ones_like(M_wx),
                                       create_graph=True, retain_graph=True)[0]
        dMwy_dt = torch.autograd.grad(M_wy, t, grad_outputs=torch.ones_like(M_wy),
                                       create_graph=True, retain_graph=True)[0]
        dMwz_dt = torch.autograd.grad(M_wz, t, grad_outputs=torch.ones_like(M_wz),
                                       create_graph=True, retain_graph=True)[0]
        dMsx_dt = torch.autograd.grad(M_sx, t, grad_outputs=torch.ones_like(M_sx),
                                       create_graph=True, retain_graph=True)[0]
        dMsy_dt = torch.autograd.grad(M_sy, t, grad_outputs=torch.ones_like(M_sy),
                                       create_graph=True, retain_graph=True)[0]
        dMsz_dt = torch.autograd.grad(M_sz, t, grad_outputs=torch.ones_like(M_sz),
                                       create_graph=True, retain_graph=True)[0]
        
        # Unscale parameters for physics equations
        k_ws_unscaled = scaler.unscale_k(k_ws)
        k_sw_unscaled = scaler.unscale_k(k_sw)
        delta_omega_s_unscaled = delta_omega_s * scaler.delta_omega_max
        delta_omega_w_unscaled = delta_omega_w * scaler.delta_omega_max
        omega1_unscaled = omega1 * scaler.omega1_max
        M_w0_unscaled = M_w0 * (scaler.M0_max - scaler.M0_min) + scaler.M0_min
        M_s0_unscaled = M_s0 * (scaler.M0_max - scaler.M0_min) + scaler.M0_min
        
        # Account for time scaling in derivatives
        time_scale = scaler.t_max
        
        # Bloch-McConnell equations (from equation 1)
        # Water pool
        residual_wx = dMwx_dt * time_scale - (
            -k_ws_unscaled * M_wx + delta_omega_w_unscaled * M_wy + 
            k_sw_unscaled * M_sx - M_wx / self.T2w
        )
        
        residual_wy = dMwy_dt * time_scale - (
            delta_omega_w_unscaled * M_wx - k_ws_unscaled * M_wy - 
            omega1_unscaled * M_wz - M_wy / self.T2w
        )
        
        residual_wz = dMwz_dt * time_scale - (
            omega1_unscaled * M_wy - k_ws_unscaled * M_wz + 
            k_sw_unscaled * M_sz + (M_w0_unscaled - M_wz) / self.T1w
        )
        
        # Solute pool
        residual_sx = dMsx_dt * time_scale - (
            k_ws_unscaled * M_wx - k_sw_unscaled * M_sx + 
            delta_omega_s_unscaled * M_sy - M_sx / self.T2s
        )
        
        residual_sy = dMsy_dt * time_scale - (
            k_ws_unscaled * M_wy + delta_omega_s_unscaled * M_sx - 
            k_sw_unscaled * M_sy - omega1_unscaled * M_sz - M_sy / self.T2s
        )
        
        residual_sz = dMsz_dt * time_scale - (
            k_ws_unscaled * M_wz + omega1_unscaled * M_sy - 
            k_sw_unscaled * M_sz + (M_s0_unscaled - M_sz) / self.T1s
        )
        
        # Mean squared error of all residuals
        physics_loss = (residual_wx**2 + residual_wy**2 + residual_wz**2 +
                       residual_sx**2 + residual_sy**2 + residual_sz**2).mean()
        
        return physics_loss
    
    def compute_ic_loss(self, k_ws, k_sw, delta_omega_s, delta_omega_w, 
                       omega1, M_w0, M_s0, scaler):
        """
        Compute initial condition loss at t=0.
        
        Initial conditions: M_w = [0, 0, M_w0], M_s = [0, 0, M_s0]
        """
        batch_size = k_ws.shape[0]
        t_zero = torch.zeros(batch_size, device=k_ws.device)
        
        # Forward pass at t=0
        M_wx, M_wy, M_wz, M_sx, M_sy, M_sz = self.forward(
            t_zero, k_ws, k_sw, delta_omega_s, delta_omega_w, omega1, M_w0, M_s0
        )
        
        # Unscale M0 values
        M_w0_unscaled = M_w0 * (scaler.M0_max - scaler.M0_min) + scaler.M0_min
        M_s0_unscaled = M_s0 * (scaler.M0_max - scaler.M0_min) + scaler.M0_min
        
        # Initial condition: transverse magnetization = 0, longitudinal = M0
        ic_loss = ((M_wx - 0)**2 + (M_wy - 0)**2 + (M_wz - M_w0_unscaled)**2 +
                   (M_sx - 0)**2 + (M_sy - 0)**2 + (M_sz - M_s0_unscaled)**2).mean()
        
        return ic_loss


# Example usage
if __name__ == "__main__":
    # Initialize scaler and model
    scaler = InputScaler(t_max=1.0, k_range=(10, 10000), 
                        delta_omega_max=1000, omega1_max=500)
    
    model = BlochMcConnellPINN(hidden_dims=[128, 128, 128, 128],
                               num_fourier_features=10,
                               fourier_scale=1.0)
    
    print("Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 100
    t = torch.rand(batch_size) * 1.0  # 0 to 1 second   
    k_ws = torch.rand(batch_size) * 9990 + 10  # 10 to 10000 Hz
    k_sw = torch.rand(batch_size) * 9990 + 10
    delta_omega_s = torch.rand(batch_size) * 2000 - 1000  # -1000 to 1000 Hz
    delta_omega_w = torch.zeros(batch_size)  # On-resonance
    omega1 = torch.rand(batch_size) * 450 + 50  # 50 to 500 Hz
    M_w0 = torch.ones(batch_size)
    M_s0 = torch.rand(batch_size) * 0.099 + 0.001  # 0.001 to 0.1
    
    # Scale inputs
    t_s, k_ws_s, k_sw_s, dos_s, dow_s, o1_s, Mw0_s, Ms0_s = scaler.scale(
        t, k_ws, k_sw, delta_omega_s, delta_omega_w, omega1, M_w0, M_s0
    )

    # TRAINING LOOP
    epochs = 1000
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("\nStarting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        M_wx, M_wy, M_wz, M_sx, M_sy, M_sz = model(
            t_s, k_ws_s, k_sw_s, dos_s, dow_s, o1_s, Mw0_s, Ms0_s
        )
        # Losses
        physics_loss = model.compute_physics_loss(
            t_s, k_ws_s, k_sw_s, dos_s, dow_s, o1_s, Mw0_s, Ms0_s, scaler
        )
        ic_loss = model.compute_ic_loss(
            k_ws_s, k_sw_s, dos_s, dow_s, o1_s, Mw0_s, Ms0_s, scaler
        )
        loss = physics_loss + ic_loss
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total loss: {loss.item():.6f} | Physics: {physics_loss.item():.6f} | IC: {ic_loss.item():.6f}")
    print("Training complete.")
    # Show final outputs
    with torch.no_grad():
        M_wx, M_wy, M_wz, M_sx, M_sy, M_sz = model(
            t_s, k_ws_s, k_sw_s, dos_s, dow_s, o1_s, Mw0_s, Ms0_s
        )
        print(f"\nFinal sample outputs: M_wz={M_wz[0].item():.4f}, M_sz={M_sz[0].item():.4f}")