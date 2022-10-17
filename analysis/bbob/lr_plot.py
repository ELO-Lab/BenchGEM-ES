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

data = {
  # Sphere (VES)
  "1_10"   : {0.5 : 1,  0.1 : 1,  0.05 : 1,  0.01 : 1,  0.005 : 1,  0.001 : 1},
  "1_100"  : {0.5 : 10, 0.1 : 10, 0.05 : 10, 0.01 : 10, 0.005 : 10, 0.001 : 10},
  "1_1000" : {0.5 : 20, 0.1 : 20, 0.05 : 20, 0.01 : 20, 0.005 : 20, 0.001 : 20},
  "1_5000" : {0.5 : 50, 0.1 : 50, 0.05 : 50, 0.01 : 50, 0.005 : 50, 0.001 : 50},

  # Rosenbrock (VES)
  "2_10"   : {0.5 : 1,  0.1 : 1,  0.05 : 1,  0.01 : 1,  0.005 : 1,  0.001 : 1},
  "2_100"  : {0.5 : 10, 0.1 : 10, 0.05 : 10, 0.01 : 10, 0.005 : 10, 0.001 : 10},
  "2_1000" : {0.5 : 20, 0.1 : 20, 0.05 : 20, 0.01 : 20, 0.005 : 20, 0.001 : 20},
  "2_5000" : {0.5 : 50, 0.1 : 50, 0.05 : 50, 0.01 : 50, 0.005 : 50, 0.001 : 50},

  # Rastrigin (GES)
  "3_10"   : {0.5 : 1,  0.1 : 1,  0.05 : 1,  0.01 : 1,  0.005 : 1,  0.001 : 1},
  "3_100"  : {0.5 : 10, 0.1 : 10, 0.05 : 10, 0.01 : 10, 0.005 : 10, 0.001 : 10},
  "3_1000" : {0.5 : 20, 0.1 : 20, 0.05 : 20, 0.01 : 20, 0.005 : 20, 0.001 : 20},
  "3_5000" : {0.5 : 50, 0.1 : 50, 0.05 : 50, 0.01 : 50, 0.005 : 50, 0.001 : 50},

  # Lunacek
  "4_10"   : {0.5 : 1,  0.1 : 1,  0.05 : 1,  0.01 : 1,  0.005 : 1,  0.001 : 1},
  "4_100"  : {0.5 : 10, 0.1 : 10, 0.05 : 10, 0.01 : 10, 0.005 : 10, 0.001 : 10},
  "4_1000" : {0.5 : 20, 0.1 : 20, 0.05 : 20, 0.01 : 20, 0.005 : 20, 0.001 : 20},
  "4_5000" : {0.5 : 50, 0.1 : 50, 0.05 : 50, 0.01 : 50, 0.005 : 50, 0.001 : 50},

  # Ackley (SGES)
  "5_10"   : {0.5 : 1,  0.1 : 1,  0.05 : 1,  0.01 : 1,  0.005 : 1,  0.001 : 1},
  "5_100"  : {0.5 : 10, 0.1 : 10, 0.05 : 10, 0.01 : 10, 0.005 : 10, 0.001 : 10},
  "5_1000" : {0.5 : 20, 0.1 : 20, 0.05 : 20, 0.01 : 20, 0.005 : 20, 0.001 : 20},
  "5_5000" : {0.5 : 50, 0.1 : 50, 0.05 : 50, 0.01 : 50, 0.005 : 50, 0.001 : 50},
}

ylim = [[0, 10], 
        [0, 100], 
        [0, 1000], 
        [0, 5000], 
        [0, 3300], 
        [0, 40000], 
        [0, 400000], 
        [0, 2000000], 
        [0, 160], 
        [0, 1400], 
        [0, 10250], 
        [0, 55000], 
        [0, 165], 
        [0, 1750], 
        [0, 17500], 
        [0, 88000], 
        [2, 5.2], 
        [3, 5.3], 
        [4.85, 5.4], 
        [5.15, 5.36]]

colors = {
  '0.5'   : 'red',
  '0.1'   : 'blue',
  '0.05'  : 'green',
  '0.01'  : 'orange',
  '0.005' : 'black',
  '0.001' : 'purple',
}

cols = ['10D', '100D', '1000D', '5000D']
rows = ['Sphere', 'Rosenbrock', 'Rastrigin', 'Lunacek', 'Ackley']

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 10))
plt.setp(axes.flat, xlabel='Iterations', ylabel='Loss')

pad = 5 # in points

for i in range(20):
  means, stds = [], []
  info_func = list(data.keys())[i]
  #if i in [0, 1, 2, 3]:
  #  algo = 'ES'
  #elif i in [4, 5, 6, 7]:
  #  algo = 'GES'
  #else:
  #  algo = 'SGES'
  algo = 'ES'
  if algo == 'ES':
    file_name = "%s_%s.npy" % (algo, info_func)
    table     = np.load("./Data/" + file_name, allow_pickle=True)
    for lr in data[info_func]:
      means.append(table.item().get(lr)[0])
      stds.append(table.item().get(lr)[1])
  else:
    for lr in data[info_func].keys():
      file_name = "%s_%s_%d.npy" % (algo, info_func, data[info_func][lr])
      table     = np.load("./Data/" + file_name, allow_pickle=True)
      means.append(table.item().get(lr)[0])
      stds.append(table.item().get(lr)[1])
  
  axes[i // 4, i % 4].set_ylim(ylim[i][0], ylim[i][1])
  axes[i // 4, i % 4].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
  plot_loss_by_iterations(axes[i // 4, i % 4], means, stds, colors)

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

fig.savefig("./Plot/" + algo + "_lr_plot" + ".png")
plt.show()
