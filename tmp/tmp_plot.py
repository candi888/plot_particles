import dataclasses
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba
from numpy.typing import NDArray

# 以下はイラレで編集可能なsvgを出力するために必要
mpl.use("Agg")
plt.rcParams["svg.fonttype"] = "none"

# フォント設定
plt.rcParams["font.family"] = "Times New Roman", "Arial"  # font familyの設定
plt.rcParams["mathtext.fontset"] = "cm"  # math fontの設定
# plt.rcParams["font.size"] = 12  # 全体のフォントサイズを変更

# 軸設定
plt.rcParams["xtick.direction"] = "out"  # x軸の目盛りの向き
plt.rcParams["ytick.direction"] = "out"  # y軸の目盛りの向き
plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
plt.rcParams["xtick.top"] = True  # x軸の上部目盛り
plt.rcParams["ytick.right"] = True  # y軸の右部目盛り
# plt.rcParams["xtick.major.pad"] = 4  # distance to major tick label in points
# plt.rcParams["ytick.major.pad"] = 3  # distance to major tick label in points
# plt.rcParams["axes.spines.top"] = False  # 上側の軸を表示するか
# plt.rcParams["axes.spines.right"] = False  # 右側の軸を表示するか
# plt.rcParams['axes.grid'] = True # グリッドの作成
# plt.rcParams['grid.linestyle']='--' #グリッドの線種

# 軸大きさ
# plt.rcParams["xtick.major.width"] = 1.0  # x軸主目盛り線の線幅
# plt.rcParams["ytick.major.width"] = 1.0  # y軸主目盛り線の線幅
# plt.rcParams["xtick.minor.width"] = 1.0  # x軸補助目盛り線の線幅
# plt.rcParams["ytick.minor.width"] = 1.0  # y軸補助目盛り線の線幅
# plt.rcParams["xtick.major.size"] = 5  # x軸主目盛り線の長さ
# plt.rcParams["ytick.major.size"] = 5  # y軸主目盛り線の長さ
# plt.rcParams["xtick.labelsize"] = 20.0  # 軸目盛のフォントサイズ変更
# plt.rcParams["ytick.labelsize"] = 20.0  # 軸目盛のフォントサイズ変更
# plt.rcParams["xtick.minor.size"] = 5  # x軸補助目盛り線の長さ
# plt.rcParams["ytick.minor.size"] = 5  # y軸補助目盛り線の長さ
# plt.rcParams["axes.linewidth"] = 1.0

# 画像保存時の余白調整など
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.05


from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.transforms import (
    Bbox,
    BboxBase,
    IdentityTransform,
    Transform,
    TransformedBbox,
    TransformedPatchPath,
    TransformedPath,
)


def main() -> None:
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    circle = ax.plot(
        [0, 0, 0.5],
        [0.5, 0, 0.6],
        "o",
        markerfacecolor="red",
        markersize=200,
        markeredgewidth=0,
        linestyle="none",
    )

    ax.grid()

    ax.set_xlim(-0.1, 0.5)
    ax.set_ylim(-0.1, 0.5)

    fig.savefig(Path(__file__).parent / "tmp2.jpeg")
    fig.savefig(Path(__file__).parent / "tmp2.svg")

    return


if __name__ == "__main__":
    main()
