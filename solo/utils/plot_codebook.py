import os
import torch
import umap
import matplotlib.pyplot as plt
import pylab
import numpy as np
import seaborn as sns
import pandas as pd
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_codebook(model, featue_bank, y_bank, args, task_id):
    umap1 = umap.UMAP(random_state=42)
    if not model.CUCL:
        plt.figure(dpi=300)
        plt.xticks([])
        plt.yticks([])
        os.makedirs('./tsne_codebook/{}'.format(args.name),exist_ok=True)
        sns.set_theme(style="darkgrid")
        with torch.no_grad():
            umap1.fit(featue_bank.cpu().numpy())
            data = umap1.transform(featue_bank.cpu().numpy())

            df = pd.DataFrame()
            df["feat_1"] = data[:, 0]
            df["feat_2"] = data[:, 1]
            df["Class"] = y_bank
            plt.figure(figsize=(9, 9))
            ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="Class",
                palette=sns.color_palette(),
                data=df,
                legend="full",
                alpha=1.0,
            )
            ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
            ax.tick_params(left=False, right=False, bottom=False, top=False)

            # X_embedded = umap1.transform(featue_bank.cpu().numpy())
            # x_min, x_max = X_embedded.min(0), X_embedded.max(0)
            # X_norm = (X_embedded - x_min) / (x_max - x_min)  # 归一化

            # for i in range(10):
            #     class_label_mask = np.where(y_bank == i)
            #     plt.scatter(X_norm[:, 0][class_label_mask], X_norm[:, 1][class_label_mask], label=str(i),s=30, c=colors[i])
            # plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig('./tsne_codebook/{}/{}.pdf'.format(args.name,task_id))
            plt.close()
    else:
        # q_x_split = torch.split(featue_bank, model.L_word, 1)

        plt.figure(dpi=300)
        plt.xticks([])
        plt.yticks([])
        os.makedirs('./tsne_codebook/{}'.format(args.name),exist_ok=True)
        sns.set_theme(style="darkgrid")
        with torch.no_grad():
            umap1.fit(featue_bank.cpu().numpy())
            data = umap1.transform(featue_bank.cpu().numpy())

            df = pd.DataFrame()
            df["feat_1"] = data[:, 0]
            df["feat_2"] = data[:, 1]
            df["Class"] = y_bank
            plt.figure(figsize=(9, 9))
            ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="Class",
                palette=sns.color_palette(),
                data=df,
                legend="full",
                alpha=1.0,
            )
            ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
            ax.tick_params(left=False, right=False, bottom=False, top=False)

            # X_embedded = umap1.transform(featue_bank.cpu().numpy())
            # x_min, x_max = X_embedded.min(0), X_embedded.max(0)
            # X_norm = (X_embedded - x_min) / (x_max - x_min)  # 归一化

            # for i in range(10):
            #     class_label_mask = np.where(y_bank == i)
            #     plt.scatter(X_norm[:, 0][class_label_mask], X_norm[:, 1][class_label_mask], label=str(i),s=30, c=colors[i])

            # plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig('./tsne_codebook/{}/{}.pdf'.format(args.name,task_id))
            plt.close()


        # for j in range(model.N_books):
        #     plt.figure(dpi=300)
        #     plt.xticks([])
        #     plt.yticks([])
        #     os.makedirs('./tsne_codebook/{}/{}'.format(args.name,j),exist_ok=True)
        #     with torch.no_grad():
        #         codebook = model.quanti_Model.C[task_id][j].cpu().numpy()

        #         umap1.fit(q_x_split[j].cpu().numpy())

        #         X_embedded = umap1.transform(q_x_split[j].cpu().numpy())
        #         x_min, x_max = X_embedded.min(0), X_embedded.max(0)
        #         X_norm = (X_embedded - x_min) / (x_max - x_min)  # 归一化

        #         C_embedded = umap1.transform(codebook)
        #         C_min, C_max = C_embedded.min(0), C_embedded.max(0)
        #         C_norm = (C_embedded - C_min) / (C_max - C_min)  # 归一化

        #         df = pd.DataFrame()
        #         df["feat_1"] = X_norm[:, 0]
        #         df["feat_2"] = X_norm[:, 1]
        #         df["Y"] = y_bank
        #         plt.figure(figsize=(9, 9))
        #         ax = sns.scatterplot(
        #             x="feat_1",
        #             y="feat_2",
        #             hue="Y",
        #             palette=sns.color_palette(),
        #             data=df,
        #             legend="full",
        #             alpha=1.0,
        #         )
        #         ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        #         ax.tick_params(left=False, right=False, bottom=False, top=False)
                

        #         # for i in range(X_norm.shape[0]):
        #         #     plt.scatter(X_norm[i, 0], X_norm[i, 1],s=2, c=colors[y_bank[i]])
        #         # plt.text(X_norm[i, 0], X_norm[i, 1], str(y_bank[i]), color=plt.cm.Set1(y_bank[i]), 
        #         #         fontdict={'weight': 'bold', 'size': 12})
        #         for i in range(C_norm.shape[0]):
        #             plt.text(C_norm[i, 0], C_norm[i, 1], str('C'), 
        #                     fontdict={'weight': 'bold', 'size': 14})
        #         plt.savefig('./tsne_codebook/{}/{}/{}.png'.format(args.name,j,task_id))
