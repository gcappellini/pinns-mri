import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, width=50, depth=4, activation='tanh', initializer='glorot_uniform'):
        super().__init__()
        
        # Select activation function
        if activation.lower() == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        layers = [nn.Linear(in_dim, width), act_fn]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act_fn]
        layers.append(nn.Linear(width, out_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(initializer)
    
    def _initialize_weights(self, initializer):
        """Initialize network weights"""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                if initializer.lower() in ['glorot_uniform', 'xavier_uniform']:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    raise ValueError(f"Unsupported initializer: {initializer}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t):
        """
        Forward pass.
        Args:
            t: time tensor (batch,) or (batch, 1)
        Returns:
            Mza, Mzb: magnetization components (batch,)
        """
        # Ensure t is 2D: (batch, 1)
        if t.dim() == 1:
            x = t.unsqueeze(-1)
        else:
            x = t
            
        out = self.net(x)
        Mza, Mzb = torch.chunk(out, 2, dim=-1)
        return Mza.squeeze(-1), Mzb.squeeze(-1)


class Longitudinal2PoolPINN(MLP):
    def __init__(self, cfg):
        super().__init__(
            in_dim=cfg.model.in_dim, 
            out_dim=cfg.model.out_dim, 
            width=cfg.model.width, 
            depth=cfg.model.depth,
            activation=cfg.model.activation,
            initializer=cfg.model.initializer
        )
        # Physics parameters (fixed)
        self.T1a = cfg.data.T1a
        self.T1b = cfg.data.T1b
        self.R1a = 1.0 / self.T1a
        self.R1b = 1.0 / self.T1b
        self.k_ab = cfg.data.k_ab
        self.k_ba = cfg.data.k_ba
        
        # Fixed initial conditions
        self.M0a = cfg.data.M0a
        self.M0b = cfg.data.M0b
        self.t_max = cfg.data.t_max

    def physics_loss(self, t_collocation):
        """Compute physics-informed loss (ODE residuals)."""
        t_collocation.requires_grad_(True)
        Mza, Mzb = self.forward(t_collocation)

        # Compute gradients dM/dt
        dMza_dt = torch.autograd.grad(
            Mza, t_collocation,
            grad_outputs=torch.ones_like(Mza),
            create_graph=True
        )[0]
        dMzb_dt = torch.autograd.grad(
            Mzb, t_collocation,
            grad_outputs=torch.ones_like(Mzb),
            create_graph=True
        )[0]

        # ODE residuals
        R1a, R1b = self.R1a, self.R1b
        k_ab, k_ba = self.k_ab, self.k_ba
        M0a, M0b = self.M0a, self.M0b

        res_a = dMza_dt + R1a * (Mza - M0a) + k_ab * Mza - k_ba * Mzb
        res_b = dMzb_dt + R1b * (Mzb - M0b) + k_ba * Mzb - k_ab * Mza

        return (res_a.pow(2) + res_b.pow(2)).mean()

    def ic_loss(self, device, n_ic):
        """Compute initial condition loss (at t≈0)."""
        eps = 1e-5
        t0 = torch.rand(n_ic, device=device) * eps
        
        Mza, Mzb = self.forward(t0)

        return ((Mza - self.M0a).pow(2) + (Mzb - self.M0b).pow(2)).mean()

    
def train_longitudinal_pinn(cfg, output_folder=""):
    """Train the PINN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Longitudinal2PoolPINN(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    
    # Loss tracking
    history = {
        'steps': [], 'loss_train': [], 'loss_test': [],
        'loss_ode_train': [], 'loss_ic_train': [],
        'loss_ode_test': [], 'loss_ic_test': []
    }
    
    best_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    t_max = cfg.data.t_max
    
    for step in range(cfg.training.num_epochs):
        # Training step
        opt.zero_grad()
        
        # Collocation points for physics loss
        t_colloc = torch.rand(cfg.training.batch_size, device=device) * t_max
        loss_ode = model.physics_loss(t_colloc)
        
        # Initial condition points
        loss_ic = model.ic_loss(device, cfg.training.n_ic)
        
        # Total loss with weights
        loss_train = cfg.training.w_ode * loss_ode + cfg.training.w_ic * loss_ic
        loss_train.backward()
        opt.step()

        # Learning rate scheduling
        if (step + 1) % cfg.training.lr_step_size == 0:
            scheduler.step()

        # Evaluation
        if step % cfg.training.eval_interval == 0:
            # Test ODE residuals
            t_test = torch.rand(cfg.training.n_test, device=device) * t_max
            t_test_grad = t_test.clone().detach().requires_grad_(True)
            loss_ode_test = model.physics_loss(t_test_grad)
            
            # Test IC loss
            loss_ic_test = model.ic_loss(device, cfg.training.n_ic)
            loss_test = cfg.training.w_ode * loss_ode_test + cfg.training.w_ic * loss_ic_test

            # Track history
            history['steps'].append(step)
            history['loss_train'].append(loss_train.item())
            history['loss_test'].append(loss_test.item())
            history['loss_ode_train'].append(loss_ode.item())
            history['loss_ic_train'].append(loss_ic.item())
            history['loss_ode_test'].append(loss_ode_test.item())
            history['loss_ic_test'].append(loss_ic_test.item())

            logger.info(f"Epoch {step}: "
                       f"loss_train={loss_train.item():.3e} (ODE={loss_ode.item():.3e}, IC={loss_ic.item():.3e}) | "
                       f"loss_test={loss_test.item():.3e} (ODE={loss_ode_test.item():.3e}, IC={loss_ic_test.item():.3e})")

            # Checkpoint best model
            if loss_test < best_loss:
                best_loss = loss_test
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = step
                logger.info(f"  -> New best: {best_loss:.6e}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {best_epoch}")

    return model, history