import os

import matplotlib
import matplotlib.collections as mcoll
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
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


def SetFrameRange_ByAllDAT(start_time: str, frame_skip: int) -> list[int]:
    int_second_starttime = int(start_time)
    print(int_second_starttime)
    for i in range(int_second_starttime, 100010, frame_skip):
        # print(f"./OUTPUT/SNAP/XUD{str(i).zfill(5)}.DAT")
        if not os.path.isfile(f"./OUTPUT/SNAP/XUD{str(i).zfill(5)}.DAT"):
            frame_range = np.arange(int_second_starttime, i, frame_skip)
            print(
                "frame_range generate break ", f"./OUTPUT/SNAP/XUD{str(i).zfill(5)}.DAT"
            )
            break
    else:
        frame_range = None  # error

    return frame_range


def GetXlimAndYlimByShoki():
    xudisa = np.loadtxt(f"./INPUT/XUDISA.DAT")
    par = xudisa[:, (0, 1, -1)]  # [x,y,disa]

    minx0, minx1 = np.amin(par[:, :], axis=0)[:2]
    maxx0, maxx1 = np.amax(par[:, :], axis=0)[:2]

    add_len_x0 = (maxx0 - minx0) / 20
    add_len_x1 = (maxx1 - minx1) / 20

    return (
        minx0 - add_len_x0,
        maxx0 + add_len_x0,
        minx1 - add_len_x1,
        maxx1 + add_len_x1,
    )


def InitPlotForFixedWall(ax, frame_skip):
    # 最初のプロットを使って差分更新と更新しないプロットを分ける
    xud = np.loadtxt(f"./OUTPUT/SNAP/XUD{str(frame_skip*1).zfill(5)}.DAT")
    tmd = np.loadtxt(f"./OUTPUT/SNAP/TMD{str(frame_skip*1).zfill(5)}.DAT")

    par = xud[:, (0, 1, 5)]  # [x,y,disa]
    nump = len(par)

    color = ["aqua", "rosybrown", "brown", "black", "violet", "magenta"]
    vector = np.vectorize(np.int_)
    par_color_idx = vector(tmd[:, (1)])  # [color]
    par_color = np.array([color[i] for i in par_color_idx])

    # ボトルネック
    circles = [
        patches.Circle((par[i, 0], par[i, 1]), par[i, -1] / 2)
        for i in range(len(par_color))
    ]
    idx_ax = set([i for i in range(nump) if par_color_idx[i] in {1, 2}])
    idx_sabun = set([i for i in range(nump) if par_color_idx[i] not in {1, 2}])
    circles_ax = [circles[i] for i in range(nump) if i in idx_ax]
    circles_sabun = [circles[i] for i in range(nump) if i in idx_sabun]
    color_ax = [par_color[i] for i in range(nump) if i in idx_ax]
    color_sabun = [par_color[i] for i in range(nump) if i in idx_sabun]

    # Collectionを作成して円を追加
    collection_ax = mcoll.PatchCollection(circles_ax, color=color_ax, linewidth=0)
    collection_sabun = mcoll.PatchCollection(
        circles_sabun, color=color_sabun, linewidth=0
    )

    # サブプロットにCollectionを追加
    ax.add_collection(collection_ax)
    sabun = ax.add_collection(collection_sabun)

    return sabun


def GetParDat(cur_time):
    # xud = np.loadtxt(rf"./OUTPUT/SNAP/XUD{cur_time:05}.DAT", usecols=(0, 1, 4, 5))

    xud = np.loadtxt(rf"./OUTPUT/SNAP/XUD{cur_time:05}.DAT", usecols=(0, 1, 4, 5))
    print(xud.shape)

    x = xud[:, 0]
    y = xud[:, 1]
    r = xud[:, 3] / 2
    p = xud[:, 2]
    nump = len(x)

    assert nump == len(y) and nump == len(r) and nump == len(p)

    return x, y, r, p, nump


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


def PlotBypatchcollection(ax, x, y, r, p, par_color, nump):
    circles = [patches.Circle((x[i], y[i]), r[i]) for i in range(nump)]

    # Collectionを作成して円を追加
    # 個々のオブジェクトのcolorを上書きしてしまうので注意
    collection = mcoll.PatchCollection(
        circles, match_original=True, facecolor=par_color, linewidth=0
    )

    # サブプロットにCollectionを追加
    ax.add_collection(collection)


def CalcSizeForScatter(fig, ax, r, maxx, minx):
    ppi = 72
    ax_size_inch = ax.figure.get_size_inches()
    ax_w_inch = ax_size_inch[0] * (
        ax.figure.subplotpars.right - ax.figure.subplotpars.left
    )
    ax_w_px = ax_w_inch * fig.get_dpi()
    size = 2 * r[:] * (ax_w_px / (maxx - minx)) * (ppi / fig.dpi)

    return size


def PlotByScatter(fig, ax, x, y, r, par_color, maxx, minx):
    size = CalcSizeForScatter(fig, ax, r, maxx, minx)
    ax.scatter(x[:], y[:], linewidths=0, s=size**2, c=par_color[:])


def GetParColorByPressureContour(p, cmap, norm):
    par_color = cmap(norm(p))

    return par_color


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


#! 1フレームごとの描画の部分
def update(frame, fig, ax, frame_skip, minx, maxx, miny, maxy, cmap, norm):
    cur_time = frame * frame_skip  # ms
    plt.cla()

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel(r"$x \mathrm{(m)}$")
    ax.set_ylabel(r"$y \mathrm{(m)}$")
    ax.minorticks_on()
    ax.set_title(rf"$t=$ {round(cur_time*1E-3,3):.2f}s")

    x, y, r, p, nump = GetParDat(cur_time)

    #! 色付け方法選択
    # par_color = GetParColorByMove(cur_time)
    par_color = GetParColorByPressureContour(p, cmap, norm)
    # par_color = GetParColorByBconForHakokeisoku(cur_time)

    #! Check Coloring Porous Area By Specific Color
    # ChangeColorInPorousArea(x, y, par_color, nump)

    #! Check Coloring WallPar Black
    ChangeColorOfWallPar(cur_time, par_color, change_color="black")
    ChangeColorOfDummyPar(cur_time, par_color, change_color="black")

    PlotByScatter(fig, ax, x, y, r, par_color, maxx, minx)

    #! SnapShot
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(
        f"{savedirname}/snap_shot/snap_{str(frame*frame_skip).zfill(5)}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    print(f"{frame_skip*frame/1000} finished")
    return


def MakeSnap(fig, ax, frame, cur_time, frame_skip, minx, maxx, miny, maxy, cmap, norm):
    # cur_time = frame * frame_skip  # ms
    plt.cla()

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel(r"$x \mathrm{(m)}$")
    ax.set_ylabel(r"$y \mathrm{(m)}$")
    ax.minorticks_on()
    ax.set_title(rf"$t=$ {round(cur_time*1E-3,3):.2f}s")

    x, y, r, p, nump = GetParDat(cur_time)

    #! 色付け方法選択
    # par_color = GetParColorByMove(cur_time)
    par_color = GetParColorByPressureContour(p, cmap, norm)
    # par_color = GetParColorByBconForHakokeisoku(cur_time)

    #! Check Coloring Porous Area By Specific Color
    # ChangeColorInPorousArea(x, y, par_color, nump)

    #! Check Coloring WallPar Black
    ChangeColorOfWallPar(cur_time, par_color, change_color="black")
    ChangeColorOfDummyPar(cur_time, par_color, change_color="black")

    PlotByScatter(fig, ax, x, y, r, par_color, maxx, minx)

    #! SnapShot
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(
        f"{savedirname}/snap_shot/snap_{str(cur_time).zfill(5)}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    print(f"{cur_time/1000} finished")
    plt.cla()

    return


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
        cwd=f"{savedirname}/snap_shot",
    )


savedirname = "./plot_output_tmp"
INF = 1 << 61 - 1


#!　main部分
def main() -> None:
    mplstyle.use("fast")

    # 出力用ディレクトリ作成
    if not os.path.exists(f"{savedirname}"):
        os.mkdir(f"{savedirname}")
    if not os.path.exists(f"{savedirname}/snap_shot"):
        os.mkdir(f"{savedirname}/snap_shot")
    if not os.path.exists(f"{savedirname}/animation"):
        os.mkdir(f"{savedirname}/animation")

    #! datファイルの00050のとこ.　SNAPの出力時間間隔が違う場合に注意
    frame_skip = 50

    #! SNAP図示の開始時間 [ms]
    start_time = f"{50:05}"
    #! Check 描画する時間範囲
    frame_range = list(range(1, 3))
    # frame_range = SetFrameRange_ByAllDAT(start_time, frame_skip)[:2]
    frame_range = SetFrameRange_ByAllDAT(start_time, frame_skip)
    # if frame_range == None:
    #     print("frame_range = None")
    #     assert False
    # if frame_range == []:
    #     print("frame_range = []")
    #     assert False

    print(frame_range)
    print(f"animation range is: {frame_range[0]/1000}[s] ~ {frame_range[-1]/1000}[s]")

    #! Check dpi
    dpi = 100

    fig = plt.figure(dpi=dpi)
    plt.clf()

    print("dpi is: ", dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    #! Check savefilename,　gifとmp4に対応
    savefilename = ""
    savefilename = f"{os.path.basename(os.getcwd())}"
    print("savefilename is: ", savefilename)

    #! 圧力Contour図の場合---------------------------------------
    #! 任意のカラーマップを選択
    cmap = matplotlib.colormaps.get_cmap("rainbow")

    #! 圧力コンターの最小，最大値設定
    minp_for_coloring = 0
    maxp_for_coloring = 1200

    norm = Normalize(vmin=minp_for_coloring, vmax=maxp_for_coloring)

    #! 手動で切り替え．．．
    PlotColorBar(ax, norm, cmap)

    #!----------------------------------------------------------

    # minx, maxx, miny, maxy = 0, 0.89, 0, 0.4

    #! ポーラスエリア拡大した範囲
    minx, maxx, miny, maxy = -0.2, 0.8, -0.02, 0.4

    # minx, maxx, miny, maxy = GetXlimAndYlimByShoki()
    print("minx, maxx, miny, maxy: ", minx, maxx, miny, maxy)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for frame, cur_time in enumerate(frame_range):
        MakeSnap(
            fig, ax, frame + 1, cur_time, frame_skip, minx, maxx, miny, maxy, cmap, norm
        )

    MakeAnimation(savefilename, start_time)

    plt.close(fig)
    print("描画終了")


if __name__ == "__main__":
    main()
