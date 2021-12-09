import spacy
import gzip
import argparse
import csv

import itertools
import numpy as np

# coo形式を利用するため
import scipy

# ヒートマップの描画のため
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォントのため
import seaborn as sns


def draw_heatmap(att_mat, name_index, axis_name):
    '''
    ヒートマップ作成関数
    Parameters
    ------------
    att_mat
        アテンション行列
    name_index : int
        文番号
    axis_name : list
        軸名。x軸・y軸ともに共通。
    '''
    plt.figure(figsize = (int(len(axis_name)/2), int(len(axis_name)/2)))

    sns.heatmap(att_mat, square=True, xticklabels=axis_name, yticklabels=axis_name, linewidths=1)
    plt.savefig('./heatmap/heatmap_{}.png'.format(name_index))
    plt.close()


def phrasebyphrase(args):

    print('loading data...')
    with open(args.src) as f:
        src_list = f.readlines()
    with open(args.trg) as f:
        trg_list = f.readlines()
    with open(args.adj) as f:
        adj_list = f.readlines()

    for i, (src, trg, adj) in enumerate(zip(src_list, trg_list, adj_list)):
        src = src.strip().split()
        trg = trg.strip().split()
        adj = adj.strip().split()

        sep_index = src.index('<sep>')

        adj_mat = np.zeros((len(src), len(src)))

        for num_num in adj:
            num0 = int(num_num.split('-')[0])
            num1 = int(num_num.split('-')[1])
            adj_mat[num0][num1] = 1


        # src1内のセルフアテンションを追加
        adj_mat[:sep_index, :sep_index] = np.ones((sep_index,sep_index))

        # src2内のセルフアテンションを追加
        adj_mat[sep_index + 1:, sep_index + 1:] = np.ones((len(src) - sep_index - 1, len(src) - sep_index - 1))

        # adjの対角成分を1に
        for j in range(len(src)):
            adj_mat[j,j] = 1

        # sepトークンのアテンションを追加
        adj_mat[:, sep_index] = np.ones(len(src))
        adj_mat[sep_index, :] = np.ones(len(src))

        # print(f'{src=}')
        # print(f'{trg=}')
        # print(f'{adj=}')
        # print(f'{sep_index=}')
        # print(f'{adj_mat=}')
        if i == 261:
            draw_heatmap(adj_mat, i, src)
            break





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src')
    parser.add_argument('-t', '--trg')
    parser.add_argument('-a', '--adj')

    args = parser.parse_args()
    phrasebyphrase(args)
    
main()