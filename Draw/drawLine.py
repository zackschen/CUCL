import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from computeAA import getAccs
plt.switch_backend('agg')
import re

# ablation study
lineType = {'Finetune':'-.','CUCL':'-'}
marker = {'0':'v',
        '1':'o',
        '2':'1',
        '3':'s',
        '4':'*',
        '5':'^'}
colors = {'0':(114/255.0,158/255.0,206/255.0),\
        '1':(255/255.0,158/255.0,74/255.0),\
        '2':(237/255.0,102/255.0,93/255.0),\
        '3':(173/255.0,139/255.0,201/255.0),\
        '4':(103/255.0,191/255.0,92/255.0),\
        '5':(249/255.0,87/255.0,56/255.0),    }

Paths = {"Finetune":"./wandb/run-20221025_154626-1oxwhomp/files/media/table/AA9_4769_affa7b457060ad4a2f33.table.json",
        "CUCL":"wandb/run-20221024_143000-3uwv8egl/files/media/table/AA9_4769_0338483f39dc24ee4774.table.json"}

plt.rc('font', family='Times New Roman',weight='normal',size=15)
figure = plt.figure(dpi=300)
a = figure.add_subplot(111) 
ax = plt.gca()
Task_num = 4
for key,path in Paths.items():
        Accs = np.array(getAccs(path))
        lengens = []
        for i in range(Task_num):
                list_task = Accs[i:,i]
                lengens.append(str(i))
                if i == 0:
                        if key == 'Finetune':
                                line1, = a.plot(range(i,10),list_task,lineType[key],color=colors[str(i)])
                        else:
                                line2, = a.plot(range(i,10),list_task,lineType[key],color=colors[str(i)])
                else:
                        a.plot(range(i,10),list_task,lineType[key],color=colors[str(i)])
# for key,path in Paths.items():
#     lines = []
#     with open(path, 'r') as read:
#         while True:
#             line = read.readline()
#             lines.append(line)
#             if not line:
#                 break
#     lines = lines[-17:-7]
#     acc_matrix=np.zeros((10,10))
#     for line_index,line in enumerate(lines):
#         accs = re.findall(r"\d+\.?\d*",line)
#         for acc_index,acc in enumerate(accs):
#             acc_matrix[line_index,acc_index] = float(acc)

#     print(acc_matrix)

#     acc = acc_matrix[-1].mean()
#     bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
#     print('Final Acc: {:5.2f}%, BWT: {:5.2f}%'.format(acc,bwt))

#     Accs = []
#     BWTs = []
#     for i in range(1,len(acc_matrix)+1):
#         matrix = acc_matrix[:i,:i]
#         acc = matrix[-1].mean()
#         bwt=np.mean((matrix[-1]-np.diag(matrix))[:-1]) 
#         Accs.append(acc)
#         BWTs.append(bwt)
#         print('Acc: {:5.2f}, BWT: {:5.2f}'.format(acc,bwt))

#     plt.plot(range(0,10),Accs,lineType,marker=marker[key],color=colors[key])
    # plt.plot(range(0,10),BWTs,lineType,marker=marker[key],color=colors[key])

# y_major_locator= MultipleLocator(10)
# ax.yaxis.set_major_locator(y_major_locator)
# yminorLocator = MultipleLocator(2)
# ax.yaxis.set_minor_locator(yminorLocator)

# x_major_locator= MultipleLocator(1)
# ax.xaxis.set_major_locator(x_major_locator)
# xminorLocator = MultipleLocator(0.5)
# ax.xaxis.set_minor_locator(xminorLocator)

# ax.xaxis.grid(True, which='minor', alpha=0.6,linewidth=0.5)
# ax.xaxis.grid(True, which='major', alpha=0.6,linewidth=1.0)
# ax.yaxis.grid(True, which='minor', alpha=0.6,linewidth=0.5)
# ax.yaxis.grid(True, which='major', alpha=0.6,linewidth=1.0)

# ax.yaxis.set_major_locator(plt.NullLocator()) 
# ax.xaxis.set_major_formatter(plt.NullFormatter()) 
# plt.ylim(50,100)
plt.xlabel('Task')
plt.ylabel('Accuracy (%)')

xticks = list(range(0,10,2))
labels = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]
xlabels=[labels[x] for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
l1 = a.legend([line1,line2], Paths.keys(), loc='lower left',ncol=2, bbox_to_anchor=[0.125,-0.28])
a.legend(lengens, loc='lower left',ncol=5, bbox_to_anchor=[0.03,-0.38])
figure.gca().add_artist(l1)
plt.grid(alpha=0.6, linestyle=':')
plt.savefig("./graph/NYG.pdf", bbox_inches='tight',dpi=300)
# plt.savefig("./graph/ablation.png", bbox_inches='tight',dpi=300)
plt.show()
pylab.show()

# import seaborn as sns
# sns.set_theme(style="darkgrid")

# # Load an example dataset with long-form data
# fmri = sns.load_dataset("fmri")

# # Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=Accs)

# plt.show()


# lineType = '-.'
# marker = {'0.5':'v','0.6':'o','0.7':'1','0.8':'s','0.9':'*','1.0':'^'}
# colors = {'0.5':(114/255.0,158/255.0,206/255.0),\
#         '0.6':(255/255.0,158/255.0,74/255.0),\
#         '0.7':(237/255.0,102/255.0,93/255.0),\
#         '0.8':(173/255.0,139/255.0,201/255.0),\
#         '0.9':(103/255.0,191/255.0,92/255.0),\
#         '1.0':(249/255.0,87/255.0,56/255.0),    }

# Paths = {'0.5':"/home/|*|/Code/gpm/checkpoints/CIFA100/Final_old5/10tasks_64batch___T_0.5_Global_0.0_local_1.0_lr_0.04_weak_False_Memory_False_Temp_0.5_Supcon_0.1/log.txt",
# '0.6':"/home/|*|/Code/gpm/checkpoints/CIFA100/Final_old5/10tasks_64batch___T_0.6_Global_0.0_local_1.0_lr_0.04_weak_False_Memory_False_Temp_0.5_Supcon_0.1/log.txt",
# '0.7':"/home/|*|/Code/gpm/checkpoints/CIFA100/Final_old5/10tasks_64batch___T_0.7_Global_0.0_local_1.0_lr_0.04_weak_False_Memory_False_Temp_0.5_Supcon_0.1/log.txt",
# '0.8':"/home/|*|/Code/gpm/checkpoints/CIFA100/Final_old5/10tasks_64batch___T_0.8_Global_0.0_local_1.0_lr_0.04_weak_False_Memory_False_Temp_0.5_Supcon_0.1/log.txt",
# '0.9':"/home/|*|/Code/gpm/checkpoints/CIFA100/Final_old5/10tasks_64batch___T_0.9_Global_0.0_local_1.0_lr_0.04_weak_False_Memory_False_Temp_0.5_Supcon_0.1/log.txt",
# '1.0':"/home/|*|/Code/gpm/checkpoints/CIFA100/Final_old5/10tasks_64batch___T_1.0_Global_0.0_local_1.0_lr_0.04_weak_False_Memory_False_Temp_0.5_Supcon_0.1/log.txt"}
# plt.rc('font', family='Times New Roman',weight='normal',size=15)
# plt.figure(dpi=300)
# ax = plt.gca()
# for key,path in Paths.items():
#     lines = []
#     with open(path, 'r') as read:
#         while True:
#             line = read.readline()
#             lines.append(line)
#             if not line:
#                 break
#     lines = lines[-17:-7]
#     acc_matrix=np.zeros((10,10))
#     for line_index,line in enumerate(lines):
#         accs = re.findall(r"\d+\.?\d*",line)
#         for acc_index,acc in enumerate(accs):
#             acc_matrix[line_index,acc_index] = float(acc)

#     print(acc_matrix)

#     acc = acc_matrix[-1].mean()
#     bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
#     print('Final Acc: {:5.2f}%, BWT: {:5.2f}%'.format(acc,bwt))

#     Accs = []
#     BWTs = []
#     for i in range(1,len(acc_matrix)+1):
#         matrix = acc_matrix[:i,:i]
#         acc = matrix[-1].mean()
#         bwt=np.mean((matrix[-1]-np.diag(matrix))[:-1]) 
#         Accs.append(acc)
#         BWTs.append(bwt)
#         print('Acc: {:5.2f}, BWT: {:5.2f}'.format(acc,bwt))

#     plt.plot(range(0,10),Accs,lineType,marker=marker[key],color=colors[key])
#     # plt.plot(range(0,10),BWTs,lineType,marker=marker[key],color=colors[key])

# # y_major_locator= MultipleLocator(10)
# # ax.yaxis.set_major_locator(y_major_locator)
# # yminorLocator = MultipleLocator(2)
# # ax.yaxis.set_minor_locator(yminorLocator)

# # x_major_locator= MultipleLocator(1)
# # ax.xaxis.set_major_locator(x_major_locator)
# # xminorLocator = MultipleLocator(0.5)
# # ax.xaxis.set_minor_locator(xminorLocator)

# # ax.xaxis.grid(True, which='minor', alpha=0.6,linewidth=0.5)
# # ax.xaxis.grid(True, which='major', alpha=0.6,linewidth=1.0)
# # ax.yaxis.grid(True, which='minor', alpha=0.6,linewidth=0.5)
# # ax.yaxis.grid(True, which='major', alpha=0.6,linewidth=1.0)

# # ax.yaxis.set_major_locator(plt.NullLocator()) 
# # ax.xaxis.set_major_formatter(plt.NullFormatter()) 
# # plt.ylim(50,100)
# plt.xlabel('Task')
# plt.ylabel('Accuracy (%)')
# plt.xticks(range(0,10), ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"])
# plt.legend(Paths.keys(), loc='lower left',ncol=3, bbox_to_anchor=[-0.12,-0.38])
# plt.grid(alpha=0.6, linestyle=':')
# plt.savefig("./graph/threshold.png", bbox_inches='tight',dpi=300)
# # plt.savefig("./graph/ablation.png", bbox_inches='tight',dpi=300)
# plt.show()
# pylab.show()