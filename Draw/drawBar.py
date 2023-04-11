import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from computeAA import getAccs
plt.switch_backend('agg')
import re
import seaborn as sns
# ablation study
lineType = {'Finetune':'-.','CUCL':'-'}
marker = {'0':'v',
        '1':'o',
        '2':'1',
        '3':'s',
        '4':'*',
        '5':'^'}
colors = {'Finetune':(114/255.0,158/255.0,206/255.0),\
        'CUCL':(255/255.0,158/255.0,74/255.0),\
        '2':(237/255.0,102/255.0,93/255.0),\
        '3':(173/255.0,139/255.0,201/255.0),\
        '4':(103/255.0,191/255.0,92/255.0),\
        '5':(249/255.0,87/255.0,56/255.0),    }

Paths = {"Finetune":"./wandb/run-20221105_105805-2t5dh3ix/files/media/table/AA9_4769_5981ff65b9bd485ef50f.table.json",
        "CUCL":"./wandb/run-20221105_105600-11v3k2vd/files/media/table/AA9_4769_e34283a0c8628d186e67.table.json"}

plt.rc('font', family='Times New Roman',weight='normal',size=15)
figure = plt.figure(dpi=300)
a = figure.add_subplot(111) 
sns.set_theme(style="ticks")
ax = plt.gca()
Task_num = 1
width = 0.08
Finetune_Acc = []
CUCL_Acc = []
for key,path in Paths.items():
    Accs = np.array(getAccs(path))
    x1 = 1
    x2 = 1
    for i in range(Task_num):
        list_task = Accs[i:,i]
        if key == "Finetune":
            Finetune_Acc.append(list_task)
        else:
            CUCL_Acc.append(list_task)

for i in range(Task_num):
    x = i
    for j in range(len(Finetune_Acc[i])):
        plt.bar(j,CUCL_Acc[i][j],color='#0081A7',edgecolor='#264653')
        plt.bar(j,Finetune_Acc[i][j],color='#F07167',edgecolor='#264653')
        x+=width
ax = plt.gca()
ax.set_ybound([40,85])
a.legend(['Simsiam w/ CUCL','Simsiam w/o CUCL'], loc='lower right')
xticks = list(range(0,10))
labels = ["Task1","Task2","Task3"]
xlabels=[str(x+1) for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
plt.ylabel('Accuracy (%)')
plt.xlabel('Train after Task')
plt.show()
plt.savefig("./graph/NYG_bar.pdf", bbox_inches='tight',dpi=300)