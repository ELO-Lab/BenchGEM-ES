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

lr = 0.1

ylim = [[0, 10],
        [0, 100], 
        [200, 1000], 
        [1500, 5000], 
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
        [4.7, 5.5], 
        [5. + .25/2, 5.5]
        ]

data = {
  "1_10"   : {1 : lr, 10 : lr},
  "1_100"  : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "1_1000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "1_5000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},

  "2_10"   : {1 : lr, 10 : lr},
  "2_100"  : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "2_1000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "2_5000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},

  "3_10"   : {1 : lr, 10 : lr},
  "3_100"  : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "3_1000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "3_5000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},

  "4_10"   : {1 : lr, 10 : lr},
  "4_100"  : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "4_1000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "4_5000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},

  "5_10"   : {1 : lr, 10 : lr},
  "5_100"  : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "5_1000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
  "5_5000" : {1 : lr, 10 : lr, 20 : lr, 50 : lr},
}

cols = ['10D', '100D', '1000D', '5000D']
rows = ['Sphere', 'Rosenbrock', 'Rastrigin', 'Lunacek', 'Ackley']

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 10))
plt.setp(axes.flat, xlabel='Iterations', ylabel='Loss')
pad = 5 

for i in range(20):
  means, stds = [], []
  info_func = list(data.keys())[i]
  #if i in [0, 1, 2, 3]:
  #  algo = 'GES'
  #else:
  #  algo = 'SGES'
  algo = 'GES'
  for k in data[info_func].keys():
    file_name = "%s_%s_%d.npy" % (algo, info_func, k)
    table     = np.load("./Data/" + file_name, allow_pickle=True)
    means.append(table.item().get(data[info_func][k])[0])
    stds.append(table.item().get(data[info_func][k])[1])
  file_name = "%s_%s.npy" % ('ES', info_func)
  table     = np.load("./Data/" + file_name, allow_pickle=True)
  means.append(table.item().get(lr)[0])
  stds.append(table.item().get(lr)[1])
  
  if info_func[2:] == '10':
    colors = { '1' : 'red', '10' : 'blue', 'VES' : 'black',}
  else:
    colors = { '1' : 'red', '10' : 'blue', '20' : 'green', '50' : 'orange', 'VES' : 'black',}
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
fig.subplots_adjust(left=0.088, top=0.96, wspace=0.3, hspace=0.3)

fig.savefig("./Plot/" + algo + "_k_plot_" + str(lr) + ".png")
plt.show()
