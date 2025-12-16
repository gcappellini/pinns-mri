"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
import os
import matplotlib.pyplot as plt
from plot import plot_pinn_vs_gt, plot_training_history
from datetime import datetime


out_fold = os.path.join(os.path.dirname(__file__), "output_dde")
if not os.path.exists(out_fold):
    os.makedirs(out_fold)

t_max=5
T1a=3.5
T1b=3.0
R1a=1.0/T1a
R1b=1.0/T1b
kab=1.0
kba=1.0
M0a=1.0
M0b=0.0

def ode_system(x, y):
    """ODE system.
    dy1/dx = y2
    dy2/dx = -y1
    """
    # Most backends
    Mza, Mzb = y[:, 0:1], y[:, 1:]
    dMza_t = dde.grad.jacobian(y, x, i=0)
    dMzb_t = dde.grad.jacobian(y, x, i=1)
    return [dMza_t + R1a*(Mza-M0a)+kab*Mza-kba*Mzb, dMzb_t + R1b*(Mzb-M0b)+kba*Mzb-kab*Mza]


def boundary(_, on_initial):
    return on_initial


geom = dde.geometry.TimeDomain(0, t_max)
ic1 = dde.icbc.IC(geom, lambda x: M0a, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: M0b, boundary, component=1)
data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, num_test=100)

layer_size = [1] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
start_time = datetime.now()
losshistory, train_state = model.train(iterations=20000, model_save_path=f"{out_fold}/pinn_bm_model.ckpt")
end_time = datetime.now()
train_state.train_time = (end_time - start_time).total_seconds()

dde.saveplot(losshistory, train_state, issave=True,loss_fname="loss_bm.dat", train_fname="train_bm.dat", test_fname="test_bm.dat", output_dir=out_fold)
plot_training_history(losshistory).savefig(os.path.join(out_fold, 'longitudinal_2pool_training_loss.png'), dpi=300, bbox_inches='tight')

# Load GT from MATLAB
gt_path = "gt/bm_longitudinal_2pool_GT.mat"
fig_pinn_gt, error = plot_pinn_vs_gt(model, gt_path, out_fold, model="dde")
fig_pinn_gt.savefig(os.path.join(out_fold, 'pinn_vs_gt.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save statistics to txt file
with open(os.path.join(out_fold, 'error_statistics.txt'), 'w') as f:
    f.write(f"Max error Mza: {error['a'].max():.3e}\n")
    f.write(f"Mean error Mza: {error['a'].mean():.3e}\n")
    f.write(f"Max error Mzb: {error['b'].max():.3e}\n")
    f.write(f"Mean error Mzb: {error['b'].mean():.3e}\n")
    f.write(f"Training time: {train_state.train_time:.2f} seconds\n")