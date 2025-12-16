import torch
import matplotlib.pyplot as plt
from model import train_longitudinal_pinn
from plot import plot_training_history, plot_pinn_vs_gt
import os
import logging
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Get Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Configure logging
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {output_dir}")
    
    # Train model
    trained, history = train_longitudinal_pinn(cfg, output_folder=output_dir)
    
    # Save trained model
    trained_path = os.path.join(output_dir, "longitudinal_2pool_pinn.pth")
    torch.save(trained.state_dict(), trained_path)
    logger.info(f"Model saved to {trained_path}")
    
    # Plot training history
    fig_loss = plot_training_history(history)
    fig_loss.savefig(os.path.join(output_dir, 'longitudinal_2pool_training_loss.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Training loss plot saved")
    plt.show()
    
    # Evaluate on MATLAB ground truth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained.eval()
    
    # Load GT from MATLAB
    gt_path = "gt/bm_longitudinal_2pool_GT.mat"
    fig_pinn_gt, error = plot_pinn_vs_gt(trained, gt_path, output_dir)
    fig_pinn_gt.savefig(os.path.join(output_dir, 'pinn_vs_gt.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Prediction vs GT plot saved")
    plt.show()

    # Save statistics to txt file
    with open(os.path.join(output_dir, 'error_statistics.txt'), 'w') as f:
        f.write(f"Max error Mza: {error['a'].max():.3e}\n")
        f.write(f"Mean error Mza: {error['a'].mean():.3e}\n")
        f.write(f"Max error Mzb: {error['b'].max():.3e}\n")
        f.write(f"Mean error Mzb: {error['b'].mean():.3e}\n")
    
    logger.info(f"Statistics saved to {os.path.join(output_dir, 'error_statistics.txt')}")


if __name__ == "__main__":
    main()