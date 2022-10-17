import numpy as np
import matplotlib.pyplot as plt

def plot_loss_by_iterations(ax, mean_history_losses, std_history_losses, colors):
    name  = list(colors.keys())
    color = list(colors.values())
    for i in range(len(mean_history_losses)):
        mean = np.array(mean_history_losses[i])
        std  = np.array(std_history_losses[i])
        ax.plot(range(mean.shape[0]), mean, label=name[i], color=color[i])
        ax.fill_between(range(std.shape[0]), mean-std, mean+std, color=color[i], alpha=0.1)
    ax.legend()

ylim = [
  [0, 5500],
  [600, 5000],
  [0, 400],
  #[0, 400]
]

algos = {
  'ES' : 'green',
  'GES' : 'blue',
  'SGES' : 'red',
  #'CMA' : 'purple',
}

task_names = [
    'HalfCheetah-v2',
    'Ant-v2',
    'Swimmer-v2',
]

cols = ['Ver1', 'Ver2', 'Ver3', 'Ver4']
rows = ['HalfCheetah-v2', 'Ant-v2', 'Swimmer-v2']

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 7))
plt.setp(axes.flat, xlabel='Iterations', ylabel='Return')
pad = 5

ver_counter = 0
for ver in [1, 2, 3, 4]:
  dir_load = './v'+str(ver)+'/aggre/'
  for task_id in range(len(rows)):
    means, stds = [], []
    for algo in algos.keys():
      file_name = algo + '_' + task_names[task_id] + ".npy"
      data      = np.load(dir_load + file_name)
      means.append(-data[0])
      stds.append(data[1])
      
    axes[task_id, ver_counter].set_ylim(ylim[task_id][0], ylim[task_id][1])
    axes[task_id, ver_counter].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes[task_id, ver_counter].grid()
    plot_loss_by_iterations(axes[task_id, ver_counter], means, stds, algos)
  ver_counter += 1


for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='medium', ha='center', va='baseline')

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='medium', ha='center', va='center', rotation=90)

fig.tight_layout()
fig.subplots_adjust(left=0.08, top=0.96, wspace=0.3, hspace=0.3)
fig.savefig("./Plot/" + "RL_plot" + ".png")
plt.show()
