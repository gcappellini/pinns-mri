import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import numpy as np

def plot_training_history(history):
    """Plot training and test loss histories with component breakdown.
    
    Args:
        history: dict with keys 'steps', 'loss_train', 'loss_test', 
                 'loss_ode_train', 'loss_ic_train', 'loss_ode_test', 'loss_ic_test'
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = history['steps']
    
    # Train losses
    ax.semilogy(steps, history['loss_train'], color='#1f77b4', label='Train Total', linewidth=2.0)
    ax.semilogy(steps, history['loss_ode_train'], color='#ff7f0e', label='Train ODE', linewidth=1.0, alpha=0.7)
    ax.semilogy(steps, history['loss_ic_train'], color='#2ca02c', label='Train IC', linewidth=1.0, alpha=0.7)
    
    # Test losses
    ax.semilogy(steps, history['loss_test'], color='#d62728', label='Test Total', linewidth=2.0, linestyle='--')
    ax.semilogy(steps, history['loss_ode_test'], color='#ff7f0e', label='Test ODE', linewidth=1.0, alpha=0.7, linestyle='--')
    ax.semilogy(steps, history['loss_ic_test'], color='#2ca02c', label='Test IC', linewidth=1.0, alpha=0.7, linestyle='--')
    
    ax.set_title('Training and Test Loss History', fontsize=13)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    return fig

def plot_pinn_vs_gt(trained, gt_path, output_dir, model="def"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_data = sio.loadmat(gt_path)
    
    # Extract time and solution
    t_gt = gt_data['GT_long']['t'][0, 0].flatten()  # (n_t,)
    M_gt = gt_data['GT_long']['M'][0, 0]  # (n_t, 2) for [Mza, Mzb]
    
    # Convert to PyTorch tensor
    t_tensor = torch.from_numpy(t_gt).float().to(device)
    M_gt_tensor = torch.from_numpy(M_gt).float().to(device)
    
    # Get PINN predictions
    if model == "def":
        with torch.no_grad():
            Mza_pinn, Mzb_pinn = trained.forward(t_tensor)
        
        Mza_pinn = Mza_pinn.cpu().numpy()
        Mzb_pinn = Mzb_pinn.cpu().numpy()

    elif model == "dde":
        with torch.no_grad():
            M_pred = trained.predict(t_tensor.unsqueeze(-1))
        
        Mza_pinn = M_pred[:, 0]
        Mzb_pinn = M_pred[:, 1]
        
    M_gt_np = M_gt_tensor.cpu().numpy()
    
    # Compute errors
    error_a = np.abs(Mza_pinn - M_gt_np[:, 0])
    error_b = np.abs(Mzb_pinn - M_gt_np[:, 1])
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Pool A and Pool B comparison
    axes[0].plot(t_gt, M_gt_np[:, 0], 'b:', linewidth=2, alpha=0.8, label='Pool A (MATLAB)')
    axes[0].plot(t_gt, Mza_pinn, 'b-', linewidth=2, alpha=0.8, label='Pool A (PINN)')
    axes[0].plot(t_gt, M_gt_np[:, 1], 'r:', linewidth=2, alpha=0.8, label='Pool B (MATLAB)')
    axes[0].plot(t_gt, Mzb_pinn, 'r-', linewidth=2, alpha=0.8, label='Pool B (PINN)')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Magnetization', fontsize=12)
    axes[0].set_title('PINN vs MATLAB Ground Truth', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Errors for both pools
    axes[1].semilogy(t_gt, error_a, 'b-', linewidth=2, alpha=0.8, label='Error Pool A')
    axes[1].semilogy(t_gt, error_b, 'r-', linewidth=2, alpha=0.8, label='Error Pool B')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Absolute Error', fontsize=12)
    axes[1].set_title('Prediction Errors', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.8, which='both')
    
    plt.tight_layout()
    return fig, {'a': error_a, 'b': error_b}
