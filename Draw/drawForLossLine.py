import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.switch_backend('agg')
from computeAA import getAccs
import re
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

# def getAccs(path):
#     results = []
#     with open(path, 'r') as f:
#         new_dict = json.loads(f.read())
#         data = new_dict['data']
#         for i,list in enumerate(data):
#             results.append(list[i])
#     return results
lineType = {'Simsiam':'-.','Simsiam w/ Diversity':'-','Simsiam w/ Quantization':'-'}
markers = {'Simsiam':'v',
        '1':'o',
        'Simsiam w/ Diversity':'1',
        'Simsiam w/ Quantization':'s',
        '4':'*',
        '5':'^'}
colors = {'Simsiam':(114/255.0,158/255.0,206/255.0),\
        'Simsiam w/ Diversity':(255/255.0,158/255.0,74/255.0),\
        'Simsiam w/ Quantization':(237/255.0,102/255.0,93/255.0),\
        '3':(173/255.0,139/255.0,201/255.0),\
        '4':(103/255.0,191/255.0,92/255.0),\
        '5':(249/255.0,87/255.0,56/255.0),    }

plt.rc('font', family='Times New Roman',weight='normal',size=15)
figure = plt.figure(dpi=300)
a = figure.add_subplot(111) 
ax = plt.gca()
sns.set_theme(style="whitegrid")

# ablation study
Paths = {"Simsiam":"./wandb/run-20221105_105805-2t5dh3ix/files/media/table/AA9_4769_5981ff65b9bd485ef50f.table.json",
        "Simsiam w/ Diversity":"wandb/run-20221110_024419-7tfy1q76/files/media/table/AA9_4293_09528f2cef11e8b967a1.table.json",
        "Simsiam w/ Quantization":"wandb/run-20221110_115625-1ibrfs02/files/media/table/AA9_4769_f41d556fff73c0e26dda.table.json"}

plt.rc('font', family='Times New Roman',weight='normal',size=15)
df = pd.DataFrame()
lengens = []
for key,path in Paths.items():
    Accs = np.array(getAccs(path))
    # print(key)
    # print(Accs)
    # print()
    MAAs =[]
    subMAAs =[]
    Task_num = len(Accs)
    for i in range(Task_num):
        subMAAs.append(Accs[i,:(i+1)].mean())
        MAAs.append(np.mean(subMAAs))
    lengens.append(key)
    line1, = a.plot(MAAs,lineType[key],color=colors[key], marker=markers[key])

xticks = list(range(0,10,2))
# ax.set_xbound(0, 9)
labels = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]
xlabels=[labels[x] for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=True, labelleft=True, grid_alpha=0.3, labelsize=9, color="#000000")
ax.grid(True, which="minor", axis="both", linestyle=":")
ax.grid(True, which="major", axis="both", linestyle="solid")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_axisbelow(True)
a.legend(lengens,loc='lower right')
plt.grid(alpha=0.6, linestyle=':')
plt.xlabel('Task')
plt.ylabel('MAA (%)')
plt.tight_layout()
plt.savefig('./graph/ForLoss.pdf')
plt.close()


