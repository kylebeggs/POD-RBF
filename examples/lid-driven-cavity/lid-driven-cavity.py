import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pod_rbf

Re = np.linspace(0, 1000, num=11)
Re[0] = 1

coords_path = "example/data/train/re-0001.csv"
x, y = np.loadtxt(
    coords_path,
    delimiter=",",
    skiprows=1,
    usecols=(1, 2),
    unpack=True,
)

# make snapshot matrix
train_snapshot = pod_rbf.mkSnapshotMatrix("example/data/train/re-%.csv")

# load validation
val = np.loadtxt(
    "example/data/validation/re-0050.csv",
    delimiter=",",
    skiprows=1,
    usecols=(0),
    unpack=True,
)


# inference the trained RBF network
model = pod_rbf.pod_rbf(energy_threshold=0.9999)
model.train(train_snapshot, Re)
sol = model.inference(50)

# plot the inferenced solution
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
cntr = ax.tricontourf(
    x, y, sol, levels=np.linspace(0, 1, num=100), cmap="viridis", extend="both"
)
fig.colorbar(cntr)

# plot the actual solution
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
cntr = ax.tricontourf(
    x, y, val, levels=np.linspace(0, 1, num=100), cmap="viridis", extend="both"
)
fig.colorbar(cntr)

# calculate and plot the percent difference between inference and actual
diff = np.nan_to_num(np.abs(sol - val) / val * 100)
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
cntr = ax.tricontourf(
    x, y, diff, levels=np.linspace(0, 100, num=100), cmap="viridis", extend="both"
)
fig.colorbar(cntr)

plt.show()
