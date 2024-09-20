import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.05


@dataclass(frozen=True)
class Dataclass_Input_Parameters:
    XUD_x_col_index: int
    XUD_y_col_index: int
    XUD_u_col_index: int
    XUD_v_col_index: int
    XUD_disa_col_index: int
    XUD_pressure_col_index: int
    TMD_move_col_index: int
    pressure_min_value_contour: int
    pressure_max_value_contour: int
    snapshot_dpi: int
    timestep_ms: int  # datファイルの00050のとこ.　SNAPの出力時間間隔が違う場合に注意
    snap_start_time_ms: int
    snap_end_time_ms: int
    xlim_min: float
    xlim_max: float
    ylim_min: float
    ylim_max: float
    cmap_for_color_contour: str


def construct_input_parameters_dataclass() -> Dataclass_Input_Parameters:
    with open("./INPUT_PARAMETERS.json", mode="r") as f:
        return Dataclass_Input_Parameters(**json.load(f))


def load_selected_par_data(snap_time_ms: int, usecols: tuple[int, ...]) -> np.ndarray:
    return np.loadtxt(rf"./OUTPUT/SNAP/XUD{snap_time_ms:05}.DAT", usecols=usecols)[:, :]


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


# TODO カス．直す
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


# Thank you chatgpt
def data_unit_to_points_size(
    diameter_in_data_units: np.ndarray, fig: plt.Figure, axis: plt.Axes
) -> np.ndarray:
    """
    todo revise
    データ単位で指定した直径を、matplotlib の scatter プロットで使用する s パラメータ（ポイントの面積）に変換します。

    Parameters
    ----------
    diameter_in_data_units : np.ndarray
        データ単位での直径の配列。
    axis : matplotlib.axes.Axes
        対象の Axes オブジェクト。

    Returns
    -------
    np.ndarray
        s パラメータとして使用可能なポイント^2 単位の面積の配列。
    """
    # データ座標からディスプレイ座標への変換関数
    trans = axis.transData.transform
    # 基準点（0,0）のディスプレイ座標
    x0, y0 = trans((0, 0))
    # 直径分離れた点のディスプレイ座標を取得（配列対応）
    x1, y1 = trans(
        np.column_stack((diameter_in_data_units, np.zeros_like(diameter_in_data_units)))
    ).T
    # ディスプレイ座標での距離（ピクセル単位）を計算
    diameter_in_pixels = np.hypot(x1 - x0, y1 - y0)
    # ピクセルをポイントに変換（1ポイント = 1/72 インチ）
    pixels_per_point = fig.dpi / 72.0
    # `s` パラメータはポイントの面積（ポイント^2）で指定
    area_in_points_squared = (diameter_in_pixels / pixels_per_point) ** 2
    return area_in_points_squared


def plot_particles_by_scatter(
    fig: plt.Figure,
    ax: plt.Axes,
    par_x: np.ndarray,
    par_y: np.ndarray,
    par_disa: np.ndarray,
    par_color: np.ndarray,
) -> None:
    # s = calc_s_for_scatter(fig=fig, ax=ax, par_disa=par_disa[:])

    s = data_unit_to_points_size(diameter_in_data_units=par_disa[:], fig=fig, axis=ax)
    ax.scatter(par_x[:], par_y[:], s=s[:], c=par_color[:], linewidths=0)

    return


# TODO よくないので修正
def get_norm_for_color_contour(physics_name: str) -> Normalize:
    if physics_name == "pressure":
        for_coloring_min = INPUT_PARAMETERS.pressure_min_value_contour
        for_coloring_max = INPUT_PARAMETERS.pressure_max_value_contour

    return Normalize(vmin=for_coloring_min, vmax=for_coloring_max)


def set_facecolor_by_physics_contour(
    physics_name: str, par_physics: np.ndarray
) -> tuple[np.ndarray, Normalize, matplotlib.colors.Colormap]:
    cmap = matplotlib.colormaps.get_cmap(INPUT_PARAMETERS.cmap_for_color_contour)
    norm = get_norm_for_color_contour(physics_name=physics_name)
    par_color = cmap(norm(par_physics[:]))

    return par_color[:], norm, cmap


def plot_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    physics_name: str,
) -> None:
    norm = get_norm_for_color_contour(physics_name=physics_name)
    mappable = ScalarMappable(cmap=INPUT_PARAMETERS.cmap_for_color_contour, norm=norm)

    #! カラーバーの軸刻み
    ticks = np.linspace(norm.vmin, norm.vmax, 5)
    ticks = mticker.LinearLocator(numticks=5)

    #! 大きさと縦向き，横向き
    fig.colorbar(
        mappable,
        ax=ax,
        ticks=ticks,
        shrink=0.6,
        orientation="horizontal",
        pad=0.1,
    ).set_label(rf"{physics_name} (Pa)")

    return


def set_ax_ticks(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.set_xlim(INPUT_PARAMETERS.xlim_min, INPUT_PARAMETERS.xlim_max)
    ax.set_ylim(INPUT_PARAMETERS.ylim_min, INPUT_PARAMETERS.ylim_max)


def set_ax_labels(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$x \mathrm{(m)}$")
    ax.set_ylabel(r"$y \mathrm{(m)}$")


def set_ax_title(ax: plt.Axes, snap_time_ms: int) -> None:
    ax.set_title(rf"$t=$ {snap_time_ms/1000:.03f}s")


# TODO 色だけでグループ分けするか，色の配列作って最後にgroupby
# TODO svg出力は一番最後にax.axis("off")とかやって出力するか？　<- scatterの大きさの計算大丈夫？
def make_snap_contour(
    fig: plt.Figure,
    ax: plt.Axes,
    snap_time_ms: int,
    save_dir_of_snap_path: Path,
    physics_col_index: int,
    physics_name: str,
) -> None:
    set_ax_ticks(ax=ax)
    set_ax_labels(ax=ax)
    set_ax_title(ax=ax, snap_time_ms=snap_time_ms)

    # TODO キャンバス更新
    if snap_time_ms == INPUT_PARAMETERS.snap_start_time_ms:
        fig.canvas.draw()

    tmp_par_data = load_selected_par_data(
        snap_time_ms=snap_time_ms,
        usecols=(
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

    par_color, norm, cmap = set_facecolor_by_physics_contour(
        physics_name=physics_name, par_physics=par_physics[:]
    )

    #! Check Coloring WallPar Black
    # ChangeColorOfWallPar(cur_time, par_color, change_color="black")
    # ChangeColorOfDummyPar(cur_time, par_color, change_color="black")

    # plot_colorbar(fig=fig, ax=ax, norm=norm, cmap=cmap, physics_name=physics_name)
    plot_particles_by_scatter(
        fig=fig,
        ax=ax,
        par_x=par_x[:],
        par_y=par_y[:],
        par_disa=par_disa[:],
        par_color=par_color[:],
    )

    fig.savefig(
        save_dir_of_snap_path / Path(f"snap{snap_time_ms:05}_{physics_name}.jpeg"),
    )

    print(f"{snap_time_ms:05} ms contour:{physics_name} plot finished")
    plt.cla()

    return


def make_snap_contour_all_time(
    snap_time_array_ms: np.ndarray,
    save_dir_physics_path: Path,
    contourphysics_col_index: int,
    physicsname: str,
) -> None:
    """指定した物理量のContour図のスナップショットを，指定した時刻全てで作成

    Args:
        snap_time_array_ms (np.ndarray): スナップショットを作成する時刻(ms)のndarray
        save_dir_path (Path): スナップやアニメーションを保存するディレクトリのPath
        contourphysics_col_index (int): Contour図を作成する物理量を指すindex
    """

    save_dir_of_snap_path = save_dir_physics_path / Path("snap_shot")
    print(save_dir_of_snap_path)
    save_dir_of_snap_path.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(dpi=INPUT_PARAMETERS.snapshot_dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plot_colorbar(fig=fig, ax=ax, physics_name=physicsname)

    for snap_time_ms in snap_time_array_ms:
        make_snap_contour(
            fig=fig,
            ax=ax,
            snap_time_ms=snap_time_ms,
            save_dir_of_snap_path=save_dir_of_snap_path,
            physics_col_index=contourphysics_col_index,
            physics_name=physicsname,
        )

    return


def make_animation_contour(
    snap_time_array_ms: np.ndarray, save_dir_physics_path: Path, physicsname: str
) -> None:
    import subprocess

    save_dir_animation_path = save_dir_physics_path / Path("animation")
    save_dir_animation_path.mkdir(exist_ok=True)

    for_ffmpeg = [f"file 'snap{i:05}_pressure.jpeg'" for i in snap_time_array_ms]
    save_file_forffmpeg_path = save_dir_physics_path / Path("snap_shot/for_ffmpeg.txt")
    with open(save_file_forffmpeg_path, mode="w") as f:
        for i in for_ffmpeg:
            f.write(f"{i}\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-r",
            "10",  # TODO 与え方どうする？INPARAMか
            # f"{int(1000/INPUT_PARAMETERS.timestep_ms)}",  # TODO 与え方どうする？INPARAMか
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(save_file_forffmpeg_path),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(save_dir_animation_path / Path(f"{physicsname}.mp4")),
        ],
        cwd=str(save_dir_physics_path / Path("snap_shot")),
    )

    return


INPUT_PARAMETERS = construct_input_parameters_dataclass()


def main() -> None:
    save_dir_path: Path = Path(__file__).parent / Path("plot_output_results")

    snap_time_array_ms: np.ndarray = np.arange(
        INPUT_PARAMETERS.snap_start_time_ms,
        INPUT_PARAMETERS.snap_end_time_ms + INPUT_PARAMETERS.timestep_ms,
        INPUT_PARAMETERS.timestep_ms,
    )

    colindex_to_physicsname: dict = {
        INPUT_PARAMETERS.XUD_pressure_col_index: "pressure"
    }

    print(
        f"animation range is: {snap_time_array_ms[0]/1000:.03f}[s] ~ {snap_time_array_ms[-1]/1000:.03f}[s]"
    )

    # 圧力コンターを出力
    cur_physicname = colindex_to_physicsname[INPUT_PARAMETERS.XUD_pressure_col_index]
    save_dir_physics_path = save_dir_path / Path(cur_physicname)
    make_snap_contour_all_time(
        snap_time_array_ms=snap_time_array_ms[:],
        save_dir_physics_path=save_dir_physics_path,
        contourphysics_col_index=INPUT_PARAMETERS.XUD_pressure_col_index,  # 圧力を指定
        physicsname=colindex_to_physicsname[INPUT_PARAMETERS.XUD_pressure_col_index],
    )
    make_animation_contour(
        snap_time_array_ms=snap_time_array_ms,
        save_dir_physics_path=save_dir_physics_path,
        physicsname=colindex_to_physicsname[INPUT_PARAMETERS.XUD_pressure_col_index],
    )

    print("描画終了")

    return


if __name__ == "__main__":
    main()
