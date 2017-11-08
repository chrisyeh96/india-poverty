# % matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

loss_train_dir = '../result/losses_train.npy'
loss_val_dir = '../result/losses_val.npy'

rsq_train_dir = '../result/r2s_train.npy'
rsq_val_dir = '../result/r2s_val.npy'

losses = {'train': list(np.load(loss_train_dir)), 'val':list(np.load(loss_val_dir))}
rsq = {'train': list(np.load(rsq_train_dir)), 'val':list(np.load(rsq_val_dir))}

print(losses["train"][-1])#/ 4

print(rsq["val"][-1])

plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
# plt.plot(np.array(losses["train"]) / 4, label="train")
# plt.plot(np.array(losses["val"]) / 4, label="val")

plt.plot(np.array(losses["train"]), label="train")
plt.plot(np.array(losses["val"]), label="val")

plt.title("LandSat-8 Multiband: Loss over epochs")
plt.xlabel("Epochs")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(np.array(rsq["train"]), label="train")
plt.plot(np.array(rsq["val"]), label="val")
plt.title("LandSat-8 Multiband: $R^2$ over epochs")
plt.xlabel("Epochs")
plt.legend()

# plt.show()
plt.savefig("Subplot LandSat-8 Multiband: $R^2$ over epochs")


plt.figure(figsize=(12, 6))
# plt.plot(np.array(losses["train"]) / 4, label="train")
# plt.plot(np.array(losses["val"]) / 4, label="val")
plt.plot(np.array(losses["train"]), label="train")
plt.plot(np.array(losses["val"]), label="val")
plt.title("LandSat-8 Multiband: Loss over epochs")
plt.xlabel("Epochs")
plt.legend()


# plt.show()
plt.savefig("LandSat-8 Multiband: Loss over epochs")




plt.figure(figsize=(12, 6))
plt.plot(np.array(rsq["train"]), label="train")
plt.plot(np.array(rsq["val"]), label="val")
plt.title("LandSat-8 Multiband: $R^2$ over epochs")
plt.xlabel("Epochs")
plt.legend()
# plt.show()
plt.savefig("LandSat-8 Multiband: $R^2$ over epochs")







import seaborn as sns

y_pred = '../result/epochs_60_Tue_Nov__7_18:01:09_2017_finetune_True_ypred.npy'
y_valid = '../result/epochs_60_Tue_Nov__7_18:01:09_2017_finetune_True_ytrue.npy'


# y_pred = np.load("./epochs_50_Tue_Oct_31_19-12-03_2017_finetune_True_ypred.npy")
# y_valid = np.load("./epochs_50_Tue_Oct_31_19-12-03_2017_finetune_True_ytrue.npy")
y_pred = np.load(y_pred)
y_valid = np.load(y_valid)


fig = sns.jointplot(np.log(y_valid[np.logical_and(y_valid > 0, y_pred > 0)]), 
                    np.log(y_pred[np.logical_and(y_valid > 0, y_pred > 0)]), 
                    kind="reg", size=8, marker=".")
fig.fig.set_size_inches((12, 8))
fig.ax_joint.set(ylabel="Log Predicted household expenditures [Taka]",
                 xlabel="Log Observed household expenditures [Taka]")
fig.ax_joint.set_title("Bangladesh: Multiband CNN predicted versus observed household expenditures", y=0.98);

fig.savefig("Loghousehod.png")


fig = sns.jointplot(y_valid[np.logical_and(y_valid > 0, y_pred > 0)], 
                    y_pred[np.logical_and(y_valid > 0, y_pred > 0)], 
                    kind="reg", size=8, marker=".")
fig.fig.set_size_inches((12, 8))
fig.ax_joint.set(ylabel="Predicted household expenditures [Taka]",
                 xlabel="Observed household expenditures [Taka]")
fig.ax_joint.set_title("Bangladesh: Multiband CNN predicted versus observed household expenditures", y=0.98);

fig.savefig("househod.png")

