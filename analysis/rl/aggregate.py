import os
import numpy as np

for i in [1, 2, 3, 4]:
  old_save_dir = './v'+str(i)+'/'
  algo_names   = ['ES','GES','SGES']
  task_names   = ['HalfCheetah-v2', 'Ant-v2', 'Swimmer-v2']
  seeds        = list(range(2016, 2025 + 1))

  for task in task_names:
    for algo in algo_names:
      name = algo + '_' + task
      history_losses, mean, std = [], [], []
      for seed in seeds:
        history_losses.append(np.load(old_save_dir + name + "_" + str(seed) + ".npy"))
      history_losses = np.array(history_losses)
      # Calculate mean and std of 10 seeds
      for iter in range(history_losses.shape[1]):
        mean.append(np.mean(history_losses[:, iter]))
        std.append(np.std(history_losses[:, iter]))

      # Save to .npy
      new_save_dir = './v'+str(i)+'/aggre/'
      file_name = name + ".npy"
      try:
        try:
          os.mkdir(new_save_dir)
        except FileExistsError:
          pass
        np.save(new_save_dir + file_name, [mean, std])
      except IOError:
        print("I/O error")
