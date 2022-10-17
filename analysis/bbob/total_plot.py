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
  # Sphere
  "1_10"   : {'ES' : [0, 0.5], 'GES' : [10, 0.5], 'SGES' : [10, 0.5], 'CMA' : [0, 0]}, # 10
  "1_100"  : {'ES' : [0, 0.5], 'GES' : [10, 0.5], 'SGES' : [10, 0.5], 'CMA' : [0, 0]}, # 100
  "1_1000" : {'ES' : [0, 0.5], 'GES' : [20, 0.5], 'SGES' : [20, 0.5], 'CMA' : [0, 0]}, # 1000
  "1_5000" : {'ES' : [0, 0.5], 'GES' : [50, 0.5], 'SGES' : [50, 0.5], 'CMA' : [0, 0]}, # 5000
  
  # Rosenbrock
  "2_10"   : {'ES' : [0, 0.001], 'GES' : [1, 0.001],  'SGES' : [1, 0.001],  'CMA' : [0, 0]}, # 3300
  "2_100"  : {'ES' : [0, 0.005], 'GES' : [10, 0.005], 'SGES' : [10, 0.005], 'CMA' : [0, 0]}, # 40000
  "2_1000" : {'ES' : [0, 0.05],  'GES' : [50, 0.05],  'SGES' : [50, 0.05],  'CMA' : [0, 0]}, # 400000
  "2_5000" : {'ES' : [0, 0.05],  'GES' : [50, 0.05],  'SGES' : [50, 0.05],  'CMA' : [0, 0]}, # 2000000

  # Rastrigin
  "3_10"   : {'ES' : [0, 0.01], 'GES' : [10, 0.01], 'SGES' : [10, 0.01], 'CMA' : [0, 0]}, # 125
  "3_100"  : {'ES' : [0, 0.01], 'GES' : [10, 0.01], 'SGES' : [10, 0.01], 'CMA' : [0, 0]}, # 1200
  "3_1000" : {'ES' : [0, 0.1],  'GES' : [50, 0.1],  'SGES' : [50, 0.1],  'CMA' : [0, 0]}, # 10250
  "3_5000" : {'ES' : [0, 0.1],  'GES' : [50, 0.1],  'SGES' : [50, 0.1],  'CMA' : [0, 0]}, # 55000

  # Lunacek
  "4_10"   : {'ES' : [0, 0.01], 'GES' : [10, 0.01], 'SGES' : [10, 0.01], 'CMA' : [0, 0]}, # 165
  "4_100"  : {'ES' : [0, 0.01], 'GES' : [20, 0.01], 'SGES' : [20, 0.01], 'CMA' : [0, 0]}, # 1750
  "4_1000" : {'ES' : [0, 0.1],  'GES' : [50, 0.1],  'SGES' : [20, 0.1],  'CMA' : [0, 0]}, # 17500
  "4_5000" : {'ES' : [0, 0.1],  'GES' : [50, 0.1],  'SGES' : [50, 0.1],  'CMA' : [0, 0]}, # 88000

  # Ackley
  "5_10"   : {'ES' : [0, 0.5], 'GES' : [10, 0.5], 'SGES' : [10, 0.5], 'CMA' : [0, 0]}, # 2 - 5.2
  "5_100"  : {'ES' : [0, 0.5], 'GES' : [1, 0.5],  'SGES' : [1, 0.5],  'CMA' : [0, 0]}, # 3 - 5.3
  "5_1000" : {'ES' : [0, 0.5], 'GES' : [1, 0.5],  'SGES' : [1, 0.5],  'CMA' : [0, 0]}, # 0 - 5.6
  "5_5000" : {'ES' : [0, 0.5], 'GES' : [1, 0.5],  'SGES' : [1, 0.5],  'CMA' : [0, 0]}, # 4.25 - 5.35
}

ylim = [[0, 10], 
        [0, 100], 
        [0, 1000], 
        [0, 5000], 
        [0, 3300], 
        [0, 40000], 
        [0, 400000], 
        [0, 2000000], 
        [0, 125], 
        [0, 1200], 
        [0, 10250], 
        [0, 55000], 
        [0, 165], 
        [0, 1750], 
        [0, 17500], 
        [0, 88000], 
        [2, 5.2], 
        [3, 5.3], 
        [0, 5.35], 
        [4.25, 5.35]]

colors = {
  'ES'   : 'green',
  'GES'  : 'blue',
  'SGES' : 'red',
  'CMA'  : 'purple',
}

cols = ['10D', '100D', '1000D', '5000D']
rows = ['Sphere', 'Rosenbrock', 'Rastrigin', 'Lunacek', 'Ackley']

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 10))
plt.setp(axes.flat, xlabel='Iterations', ylabel='Loss')

pad = 5 # in points

for i in range(20):
  means, stds = [], []
  info_func   = list(data.keys())[i]
  for algo in data[info_func].keys():
    if algo == 'ES':
      file_name = "%s_%s.npy" % (algo, info_func)
      table     = np.load("./Data/" + file_name, allow_pickle=True)
      means.append(table.item().get(data[info_func][algo][1])[0])
      stds.append(table.item().get(data[info_func][algo][1])[1])

    elif algo == 'CMA':
      if info_func[2:] == '5000':
        continue
      file_name = "%s_%s.npy" % (algo, info_func)
      table     = np.load("./Data/" + file_name, allow_pickle=True)
      means.append(table.item().get(1.0)[0])
      stds.append(table.item().get(1.0)[1])

    else:
      file_name = "%s_%s_%d.npy" % (algo, info_func, data[info_func][algo][0])
      table     = np.load("./Data/" + file_name, allow_pickle=True)
      means.append(table.item().get(data[info_func][algo][1])[0])
      stds.append(table.item().get(data[info_func][algo][1])[1])

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

fig.savefig("./Plot/" + "total_plot" + ".png")
plt.show()
