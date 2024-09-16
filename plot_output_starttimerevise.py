import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba


@dataclass(frozen=True)
class Dataclass_Input_Parameters:
    XUD_x_col_index: int = 0
    XUD_y_col_index: int = 1
    XUD_u_col_index: int = 2
    XUD_v_col_index: int = 3
    XUD_disa_col_index: int = 5
    XUD_pressure_col_index: int = 6
    TMD_move_col_index: int = 1
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
        return Dataclass_Input_Parameters(**json.load(f))


INPUT_PARAMETERS = construct_input_parameters_dataclass()


def get_selected_par_data(
    snap_time_ms: int, selected_cols_index: tuple[int, ...]
) -> np.ndarray:
    return np.loadtxt(
        rf"./OUTPUT/SNAP/XUD{snap_time_ms:05}.DAT", usecols=selected_cols_index
    )[:, :]


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


def calc_s_for_scatter(
    fig: plt.Figure, ax: plt.Axes, par_disa: np.ndarray
) -> np.ndarray:
    ppi = 72
    ax_size_inch = ax.figure.get_size_inches()
    ax_w_inch = ax_size_inch[0] * (
        ax.figure.subplotpars.right - ax.figure.subplotpars.left
    )
    ax_w_px = ax_w_inch * fig.dpi
    s = (
        par_disa[:]
        * (ax_w_px / (INPUT_PARAMETERS.xlim_max - INPUT_PARAMETERS.xlim_min))
        * (ppi / fig.dpi)
    ) ** 2

    return s[:]


def plot_particles_by_scatter(
    fig: plt.Figure,
    ax: plt.Axes,
    par_x: np.ndarray,
    par_y: np.ndarray,
    par_disa: np.ndarray,
    par_color: np.ndarray,
) -> None:
    s = calc_s_for_scatter(fig=fig, ax=ax, par_disa=par_disa[:])
    ax.scatter(par_x[:], par_y[:], s=s[:], c=par_color[:], linewidths=0)


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


# TODO 色だけでグループ分けするか，色の配列作って最後にgroupby
# TODO svg出力は一番最後にax.axis("off")とかやって出力するか？
def make_snap_contour(
    fig: plt.Figure,
    ax: plt.Axes,
    snap_time_ms: int,
    save_dir_of_snap_path: Path,
    physics_col_index: int,
    physics_name: str,
) -> None:
    ax.minorticks_on()
    ax.set_xlim(INPUT_PARAMETERS.xlim_min, INPUT_PARAMETERS.xlim_max)
    ax.set_ylim(INPUT_PARAMETERS.ylim_min, INPUT_PARAMETERS.ylim_max)
    ax.set_xlabel(r"$x \mathrm{(m)}$")
    ax.set_ylabel(r"$y \mathrm{(m)}$")
    ax.set_title(rf"$t=$ {snap_time_ms/1000:.3}s")

    tmp_par_data = get_selected_par_data(
        snap_time_ms=snap_time_ms,
        selected_cols_index=(
            INPUT_PARAMETERS.XUD_x_col_index,
            INPUT_PARAMETERS.XUD_y_col_index,
            INPUT_PARAMETERS.XUD_disa_col_index,
            physics_col_index,
        ),
    )[:, :]
    par_x = tmp_par_data[:, 0]
    par_y = tmp_par_data[:, 1]
    par_disa = tmp_par_data[:, 2]
    par_physics = tmp_par_data[:, 3]

    #! 色付け方法選択
    par_color = set_facecolor_by_physics_contour(
        physics_name=physics_name, par_physics=par_physics[:]
    )

    #! Check Coloring Porous Area By Specific Color
    # ChangeColorInPorousArea(x, y, par_color, nump)

    #! Check Coloring WallPar Black
    # ChangeColorOfWallPar(cur_time, par_color, change_color="black")
    # ChangeColorOfDummyPar(cur_time, par_color, change_color="black")

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plot_particles_by_scatter(
        fig=fig,
        ax=ax,
        par_x=par_x[:],
        par_y=par_y[:],
        par_disa=par_disa[:],
        par_color=par_color[:],
    )

    #! SnapShot保存
    fig.savefig(
        save_dir_of_snap_path / Path(f"snap{snap_time_ms:05}_{physics_name}.jpeg"),
        bbox_inches="tight",
        pad_inches=0.05,
    )

    print(f"{snap_time_ms:05} ms contour:{physics_name} plot finished")
    plt.cla()

    return


def make_snap_contour_all_time(
    snap_time_array_ms: np.ndarray, save_dir_path: Path, contourphysics_col_index: int
) -> None:
    """指定した物理量のContour図のスナップショットを，指定した時刻全てで作成

    Args:
        snap_time_array_ms (np.ndarray): スナップショットを作成する時刻(ms)のndarray
        save_dir_path (Path): スナップやアニメーションを保存するディレクトリのPath
        contourphysics_col_index (int): Contour図を作成する物理量を指すindex
    """
    colindex_to_physicsname = {
        INPUT_PARAMETERS.XUD_pressure_col_index: "pressure"
    }  # TODO　あとでリファクタリング
    physicsname = colindex_to_physicsname[contourphysics_col_index]

    save_dir_of_snap_path = save_dir_path / Path(physicsname) / Path("snap_shot")
    print(save_dir_of_snap_path)
    save_dir_of_snap_path.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(dpi=INPUT_PARAMETERS.snapshot_dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    for snap_time_ms in snap_time_array_ms:
        make_snap_contour(
            fig=fig,
            ax=ax,
            snap_time_ms=snap_time_ms,
            save_dir_of_snap_path=save_dir_of_snap_path,
            physics_col_index=contourphysics_col_index,
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

    snap_time_array_ms = np.arange(
        INPUT_PARAMETERS.snap_start_time_ms,
        INPUT_PARAMETERS.snap_end_time_ms + INPUT_PARAMETERS.timestep_ms,
        INPUT_PARAMETERS.timestep_ms,
    )
    print(
        f"animation range is: {snap_time_array_ms[0]/1000:.3}[s] ~ {snap_time_array_ms[-1]/1000:.3}[s]"
    )

    # 圧力コンターを出力
    make_snap_contour_all_time(
        snap_time_array_ms=snap_time_array_ms[:],
        save_dir_path=save_dir_path,
        contourphysics_col_index=INPUT_PARAMETERS.XUD_pressure_col_index,  # 圧力を指定
    )

    print("描画終了")


if __name__ == "__main__":
    main()
