import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.collections as mcoll
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba

#!フォント設定
# plt.rcParams["font.family"] = "Times New Roman"  # font familyの設定
# plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
# plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。
# # plt.rcParams['xtick.labelsize'] = 9 # 軸だけ変更されます。
# # plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
# #!日本語フォント設定
# font_paths = findSystemFonts()
# use_jp_font = [path for path in font_paths if "yumin.ttf" in path.lower()]
# print(font_paths)
# assert len(use_jp_font) == 1
# jp_font = FontProperties(fname=use_jp_font[0])
# #!軸設定
# plt.rcParams["xtick.direction"] = "out"  # x軸の目盛りの向き
# plt.rcParams["ytick.direction"] = "out"  # y軸の目盛りの向き
# # plt.rcParams['axes.grid'] = True # グリッドの作成
# # plt.rcParams['grid.linestyle']='--' #グリッドの線種
# plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
# plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加
# plt.rcParams["xtick.top"] = True  # x軸の上部目盛り
# plt.rcParams["ytick.right"] = True  # y軸の右部目盛り
# #!軸大きさ
# # plt.rcParams["xtick.major.width"] = 1.0             #x軸主目盛り線の線幅
# # plt.rcParams["ytick.major.width"] = 1.0             #y軸主目盛り線の線幅
# # plt.rcParams["xtick.minor.width"] = 1.0             #x軸補助目盛り線の線幅
# # plt.rcParams["ytick.minor.width"] = 1.0             #y軸補助目盛り線の線幅
# # plt.rcParams["xtick.major.size"] = 10               #x軸主目盛り線の長さ
# # plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ
# # plt.rcParams["xtick.minor.size"] = 5                #x軸補助目盛り線の長さ
# # plt.rcParams["ytick.minor.size"] = 5                #y軸補助目盛り線の長さ
# # plt.rcParams["axes.linewidth"] = 1.0                #囲みの太さ
# #!凡例設定
# plt.rcParams["legend.fancybox"] = False  # 丸角OFF
# plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
# plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
# plt.rcParams["legend.markerscale"] = 5  # markerサイズの倍率
# # * 以下，よく使うコマンドのメモ
# # .set_label(label="全断面に対する出現率", fontproperties=jp_font)

# # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
# # plt.savefig(
# #     f"{outdir}/danmen_appearrate.png",
# #     bbox_inches="tight",
# #     pad_inches=0.1,
# #     transparent="True",
# # )


@dataclass(frozen=True)
class Dataclass_Input_Parameters:
    XUD_x_row_index: int = 0
    XUD_y_row_index: int = 1
    XUD_u_row_index: int = 2
    XUD_v_row_index: int = 3
    XUD_disa_row_index: int = 5
    XUD_pressure_row_index: int = 6
    TMD_move_row_index: int = 1
    pressure_min_value_contour: int = 0
    pressure_max_value_contour: int = 2000
    snapshot_dpi: int = 100
    timestep_ms: int = (
        50  # datファイルの00050のとこ.　SNAPの出力時間間隔が違う場合に注意
    )
    snap_start_time_ms: int = 50
    snap_end_time_ms: int = 1000
    xlim_min: float = 0.0
    xlim_max: float = 115.0
    ylim_min: float = 0.0
    ylim_max: float = 115.0


def construct_input_parameters_dataclass() -> Dataclass_Input_Parameters:
    with open("./INPUT_PARAMETERS.json", mode="r") as f:
        INPUT_PARAMETERS = Dataclass_Input_Parameters(json.load(f))

    return INPUT_PARAMETERS


INPUT_PARAMETERS = construct_input_parameters_dataclass()


def SetFrameRange_ByAllDAT(start_time: str, frame_skip: int) -> list[int]:
    int_second_starttime = int(start_time)
    print(int_second_starttime)
    for i in range(int_second_starttime, 100010, frame_skip):
        # print(f"./OUTPUT/SNAP/XUD{str(i).zfill(5)}.DAT")
        if not os.path.isfile(f"./OUTPUT/SNAP/XUD{str(i).zfill(5)}.DAT"):
            snap_time_array_ms = np.arange(int_second_starttime, i, frame_skip)
            print(
                "snap_time_array_ms generate break ",
                f"./OUTPUT/SNAP/XUD{str(i).zfill(5)}.DAT",
            )
            break
    else:
        snap_time_array_ms = None  # error

    return snap_time_array_ms


def get_x_y_disa_physics(
    snap_time_ms: int, physics_row_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    par_data = np.loadtxt(
        rf"./OUTPUT/SNAP/XUD{snap_time_ms:05}.DAT", usecols=(0, 1, 4, 5)
    )

    x = par_data[:, INPUT_PARAMETERS.XUD_x_row_index]
    y = par_data[:, INPUT_PARAMETERS.XUD_y_row_index]
    disa = par_data[:, INPUT_PARAMETERS.XUD_disa_row_index]
    physics = par_data[:, physics_row_index]

    return x[:], y[:], disa[:], physics[:]


def GetParColorByMove(cur_time):
    tmd = np.loadtxt(rf"./OUTPUT/SNAP/TMD{str(cur_time).zfill(5)}.DAT")

    color = ["aqua", "rosybrown", "brown", "black", "violet", "magenta"]
    vector = np.vectorize(np.int_)
    par_color_idx = vector(tmd[:, (1)])  # [color]
    par_color = np.array([color[i] for i in par_color_idx])

    return par_color


def GetParColorByBconForHakokeisoku(cur_time):
    bcon = np.loadtxt(rf"./OUTPUT/SNAP/for_hako_keisoku{str(cur_time).zfill(5)}.DAT")
    vector = np.vectorize(np.int_)
    par_color_key = vector(bcon)

    # 0:水, 1: 自由表面, -1: ダミー, -2: 壁粒子
    color = {0: "aqua", -1: "red", -2: "black", 1: "yellow"}
    par_color = np.array([color[key] for key in par_color_key])

    return par_color


def ChangeColorInPorousArea(x, y, par_color, nump):
    porous_area_vertex = np.loadtxt(rf"./INPUT/porous_area.dat")
    left, right = porous_area_vertex[0, 0], porous_area_vertex[3, 0]
    upper, lower = porous_area_vertex[0, 1], porous_area_vertex[1, 1]

    for i in range(nump):
        curx, cury = x[i], y[i]
        if left <= curx <= right and lower <= cury <= upper:
            par_color[i] = "blue"


def ChangeColorOfWallPar(cur_time, par_color, change_color):
    tmd = np.loadtxt(rf"./OUTPUT/SNAP/TMD{str(cur_time).zfill(5)}.DAT")

    vector = np.vectorize(np.int_)
    par_color_idx = vector(tmd[:, (1)])  # [color]

    for i, pidx in enumerate(par_color_idx):
        if np.mod(pidx, 3) == 1:
            par_color[i] = to_rgba(change_color)
            # par_color[i] = (0, 0, 0, 0)  # 透明


def ChangeColorOfDummyPar(cur_time, par_color, change_color):
    tmd = np.loadtxt(rf"./OUTPUT/SNAP/TMD{str(cur_time).zfill(5)}.DAT")

    vector = np.vectorize(np.int_)
    par_color_idx = vector(tmd[:, (1)])  # [color]

    for i, pidx in enumerate(par_color_idx):
        if np.mod(pidx, 3).astype(np.int_) == 2:
            par_color[i] = to_rgba(change_color)
            # par_color[i] = (0, 0, 0, 0)  # 透明


def CalcSizeForScatter(fig: plt.Figure, ax: plt.Axes, par_disa: np.ndarray) -> float:
    ppi = 72
    ax_size_inch = ax.figure.get_size_inches()
    ax_w_inch = ax_size_inch[0] * (
        ax.figure.subplotpars.right - ax.figure.subplotpars.left
    )
    ax_w_px = ax_w_inch * fig.dpi
    size = (
        par_disa[:]
        * (ax_w_px / (INPUT_PARAMETERS.xlim_max - INPUT_PARAMETERS.xlim_min))
        * (ppi / fig.dpi)
    )

    return size


def PlotByScatter(fig, ax, par_x, par_y, par_disa, par_color, maxx, minx):
    size = CalcSizeForScatter(fig, ax, r, maxx, minx) ** 2
    ax.scatter(x[:], y[:], linewidths=0, s=size, c=par_color[:])


# TODO よくないので修正
def set_facecolor_by_physics_contour(
    physics_name: str, par_physics: np.ndarray
) -> np.ndarray:
    #! 任意のカラーマップを選択
    cmap = matplotlib.colormaps.get_cmap("rainbow")

    #! 圧力コンターの最小，最大値設定
    if physics_name == "pressure":
        for_coloring_min = INPUT_PARAMETERS.pressure_min_value_contour
        for_coloring_max = INPUT_PARAMETERS.pressure_max_value_contour

    norm = Normalize(vmin=for_coloring_min, vmax=for_coloring_max)
    #!----------------------------------------------------------

    par_color = cmap(norm(par_physics[:]))

    return par_color[:]


#!　Colorbarのいろいろな調整
def PlotColorBar(ax, norm, cmap):
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable._A = []

    #! カラーバーの軸刻み
    ticks = np.linspace(norm.vmin, norm.vmax, 5)
    ticks = mticker.LinearLocator(numticks=5)

    #! 大きさと縦向き，横向き
    plt.colorbar(
        mappable,
        ax=ax,
        ticks=ticks,
        shrink=0.62,
        orientation="horizontal",
        pad=0.1,
    ).set_label(r"Pressure [Pa]")


# 色だけでグループ分けするか，色の配列作って最後にgroupby
# svg出力は一番最後にax.axis("off")とかやって出力するか？
def make_snap_contour(
    fig: plt.Figure,
    ax: plt.Axes,
    snap_time_ms: int,
    save_dir_of_snap_path: Path,
    physics_row_index: int,
    physics_name: str,
) -> None:
    ax.set_xlim(INPUT_PARAMETERS.xlim_min, INPUT_PARAMETERS.xlim_max)
    ax.set_ylim(INPUT_PARAMETERS.ylim_min, INPUT_PARAMETERS.ylim_max)
    ax.set_xlabel(r"$x \mathrm{(m)}$")
    ax.set_ylabel(r"$y \mathrm{(m)}$")
    ax.minorticks_on()
    ax.set_title(rf"$t=$ {snap_time_ms/1000:.3}s")

    par_x, par_y, par_disa, par_physics = get_x_y_disa_physics(
        snap_time_ms=snap_time_ms, physics_row_index=physics_row_index
    )

    #! 色付け方法選択
    par_color = set_facecolor_by_physics_contour(
        physics_name=physics_name, par_physics=par_physics[:]
    )

    #! Check Coloring Porous Area By Specific Color
    # ChangeColorInPorousArea(x, y, par_color, nump)

    #! Check Coloring WallPar Black
    # ChangeColorOfWallPar(cur_time, par_color, change_color="black")
    # ChangeColorOfDummyPar(cur_time, par_color, change_color="black")

    PlotByScatter(fig, ax, x, y, r, par_color, maxx, minx)

    # #! SnapShot
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # plt.savefig(
    #     f"{savedirname}/snap_shot/snap_{str(cur_time).zfill(5)}.png",
    #     bbox_inches="tight",
    #     pad_inches=0.1,
    # )

    # print(f"{cur_time/1000} finished")
    plt.cla()

    return


def make_snap_contour_all_time(
    snap_time_array_ms: np.ndarray, save_dir_path: Path, contourphysics_row_index: int
) -> None:
    rowindex_to_physicsname = {
        INPUT_PARAMETERS.XUD_pressure_row_index: "pressure"
    }  # TODO　あとでリファクタリング

    physicsname = rowindex_to_physicsname[contourphysics_row_index]
    save_dir_of_snap_path = save_dir_path / Path(physicsname)
    fig = plt.figure(dpi=INPUT_PARAMETERS.snapshot_dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    for snap_time_ms in snap_time_array_ms:
        make_snap_contour(
            fig=fig,
            ax=ax,
            snap_time_ms=snap_time_ms,
            save_dir_of_snap_path=save_dir_of_snap_path,
            physics_row_index=contourphysics_row_index,
            physics_name=physicsname,
        )


def MakeAnimation(savefilename, start_time):
    import subprocess

    ffmpeg_path = "/LARGE1/gr10162/yamanaka/ffmpeg-release-amd64-static/ffmpeg-6.0-amd64-static/ffmpeg"
    # ffmpeg_path = "ffmpeg"

    subprocess.run(
        [
            f"{ffmpeg_path}",
            "-y",
            "-framerate",
            "200",
            "-start_number",
            f"{start_time}",
            "-i",
            "snap_%*.png",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            f"../animation/{savefilename}.mp4",
        ],
        # cwd=f"{savedirname}/snap_shot",
    )


#!　main部分
def main() -> None:
    save_dir_path = Path(__file__).parent / Path("plot_output_results")
    (save_dir_path / Path("snap_shot")).mkdir(exist_ok=True)

    snap_time_array_ms = np.arange(
        INPUT_PARAMETERS.snap_start_time_ms,
        INPUT_PARAMETERS.snap_end_time_ms + INPUT_PARAMETERS.timestep_ms,
        INPUT_PARAMETERS.timestep_ms,
    )
    print(
        f"animation range is: {snap_time_array_ms[0]/1000:.3}[s] ~ {snap_time_array_ms[-1]/1000:.3}[s]"
    )

    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # for frame, cur_time in enumerate(snap_time_array_ms):
    # MakeSnap(
    #     fig, ax, frame + 1, cur_time, frame_skip, minx, maxx, miny, maxy, cmap, norm
    # )

    # MakeAnimation(savefilename, start_time)

    print("描画終了")


if __name__ == "__main__":
    main()
