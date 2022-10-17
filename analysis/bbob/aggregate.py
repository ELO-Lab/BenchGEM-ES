import os
import numpy as np

old_save_dir = './Data/'
algo_name    = 'ES'
tail_name    = '5_1000'
seeds 	     = list(range(2016, 2025 + 1))
if algo_name == 'CMA':
    lrs = [1.0]
else:
    lrs = [0.5, 0.05, 0.005, 0.1, 0.01, 0.001]

indiviual_data_tables = []
for seed in seeds:
    indiviual_data_tables.append(np.load(old_save_dir + algo_name + "_" + str(seed) + "_" + tail_name + ".npy", allow_pickle=True))

data_table = {}
for lr in lrs:
    history_losses, mean, std = [], [], []
    for i in range(len(seeds)):
        history_losses.append(indiviual_data_tables[i].item().get(lr)[0])
    history_losses = np.array(history_losses)
    # Calculate mean and std of 10 seeds
    for iter in range(history_losses.shape[1]):
        mean.append(np.mean(history_losses[:, iter]))
        std.append(np.std(history_losses[:, iter]))
    data_table[lr] = [mean, std]

# Save to .npy
new_save_dir = './aggre/'
file_name = algo_name + "_" + tail_name + ".npy"
try:
    try:
        os.mkdir(new_save_dir)
    except FileExistsError:
        pass
    np.save(new_save_dir + file_name, data_table)
except IOError:
    print("I/O error")
