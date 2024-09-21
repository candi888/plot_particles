import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.05


@dataclass(frozen=True)
class Dataclass_Input_Parameters:
    XUD_x_col_index: int
    XUD_y_col_index: int
    XUD_u_col_index: int
    XUD_v_col_index: int
    XUD_disa_col_index: int
    TMD_move_col_index: int
    timestep_ms: int
    snap_start_time_ms: int
    snap_end_time_ms: int
    xlim_min: float
    xlim_max: float
    ylim_min: float
    ylim_max: float
    snapshot_dpi: int
    pressure_label: str
    pressure_col_index: int
    pressure_min_value_contour: int
    pressure_max_value_contour: int
    pressure_cmap: str
    pressure_wall_particle_color_rgba: list[float]
    pressure_dummy_particle_color_rgba: list[float]


def construct_input_parameters_dataclass() -> Dataclass_Input_Parameters:
    with open("./INPUT_PARAMETERS.yaml", mode="r") as f:
        return Dataclass_Input_Parameters(**yaml.safe_load(f))


def get_mask_array_by_plot_region(snap_time_ms: int) -> np.ndarray:
    ori_data = np.loadtxt(
        rf"./OUTPUT/SNAP/XUD{snap_time_ms:05}.DAT",
        usecols=(
            IN_PARAMS.XUD_x_col_index,
            IN_PARAMS.XUD_y_col_index,
            IN_PARAMS.XUD_disa_col_index,
        ),
    )

    x = ori_data[:, 0]
    y = ori_data[:, 1]
    margin = np.max(ori_data[:, 2]) * 3  # 最大粒径の３倍分marginを設定

    # 範囲条件を設定
    mask = (
        (x >= IN_PARAMS.xlim_min - margin)
        & (x <= IN_PARAMS.xlim_max + margin)
        & (y >= IN_PARAMS.ylim_min - margin)
        & (y <= IN_PARAMS.ylim_max + margin)
    )

    return mask


def load_selected_par_data(
    snap_time_ms: int, mask_array: np.ndarray, usecols: tuple[int, ...]
) -> np.ndarray:
    ori_data = np.loadtxt(rf"./OUTPUT/SNAP/XUD{snap_time_ms:05}.DAT", usecols=usecols)

    masked_data = ori_data[mask_array, :]

    return masked_data.T


def change_color_wall_particles(
    snap_time_ms: int, par_color: np.ndarray, physics_name: str, mask_array: np.ndarray
) -> np.ndarray:
    masked_move_index = np.loadtxt(
        rf"./OUTPUT/SNAP/TMD{snap_time_ms:05}.DAT",
        dtype=np.int_,
        usecols=IN_PARAMS.TMD_move_col_index,
    )[mask_array]

    mask_by_wall = np.mod(masked_move_index, 3) == 1

    par_color[mask_by_wall] = np.array(
        getattr(IN_PARAMS, f"{physics_name}_wall_particle_color_rgba")
    )

    print(par_color)
    return par_color


def change_color_dummy_particles(
    snap_time_ms: int, par_color: np.ndarray, physics_name: str, mask_array: np.ndarray
) -> np.ndarray:
    masked_move_index = np.loadtxt(
        rf"./OUTPUT/SNAP/TMD{snap_time_ms:05}.DAT",
        dtype=np.int_,
        usecols=IN_PARAMS.TMD_move_col_index,
    )[mask_array]

    mask_by_dummy = np.mod(masked_move_index, 3) == 2

    par_color[mask_by_dummy] = np.array(
        getattr(IN_PARAMS, f"{physics_name}_dummy_particle_color_rgba")
    )

    return par_color


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

    s = data_unit_to_points_size(diameter_in_data_units=par_disa, fig=fig, axis=ax)
    ax.scatter(par_x, par_y, s=s, c=par_color, linewidths=0)

    return


def get_norm_for_color_contour(physics_name: str) -> Normalize:
    return Normalize(
        vmin=getattr(IN_PARAMS, f"{physics_name}_min_value_contour"),
        vmax=getattr(IN_PARAMS, f"{physics_name}_max_value_contour"),
    )


def set_facecolor_by_physics_contour(
    physics_name: str, par_physics: np.ndarray
) -> np.ndarray:
    cmap = matplotlib.colormaps.get_cmap(getattr(IN_PARAMS, f"{physics_name}_cmap"))
    norm = get_norm_for_color_contour(physics_name=physics_name)
    par_color = cmap(norm(par_physics))

    return par_color


def plot_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    physics_name: str,
) -> None:
    norm = get_norm_for_color_contour(physics_name=physics_name)
    cmap = matplotlib.colormaps.get_cmap(getattr(IN_PARAMS, f"{physics_name}_cmap"))
    mappable = ScalarMappable(cmap=cmap, norm=norm)

    #! カラーバーの軸刻み
    assert norm.vmin is not None and norm.vmax is not None
    ticks = np.linspace(norm.vmin, norm.vmax, 5)

    #! 大きさと縦向き，横向き
    fig.colorbar(
        mappable,
        ax=ax,
        ticks=ticks,
        shrink=0.6,
        orientation="horizontal",
        pad=0.1,
    ).set_label(f"{getattr(IN_PARAMS,f"{physics_name}_label")}")

    return


def set_ax_ticks(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.set_xlim(IN_PARAMS.xlim_min, IN_PARAMS.xlim_max)
    ax.set_ylim(IN_PARAMS.ylim_min, IN_PARAMS.ylim_max)


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
    if snap_time_ms == IN_PARAMS.snap_start_time_ms:
        fig.canvas.draw()

    mask_array = get_mask_array_by_plot_region(snap_time_ms=snap_time_ms)

    par_x, par_y, par_disa, par_physics = load_selected_par_data(
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=(
            IN_PARAMS.XUD_x_col_index,
            IN_PARAMS.XUD_y_col_index,
            IN_PARAMS.XUD_disa_col_index,
            physics_col_index,
        ),
    )

    par_color = set_facecolor_by_physics_contour(
        physics_name=physics_name, par_physics=par_physics
    )

    change_color_wall_particles(
        snap_time_ms=snap_time_ms,
        par_color=par_color,
        physics_name=physics_name,
        mask_array=mask_array,
    )
    change_color_dummy_particles(
        snap_time_ms=snap_time_ms,
        par_color=par_color,
        physics_name=physics_name,
        mask_array=mask_array,
    )

    plot_particles_by_scatter(
        fig=fig,
        ax=ax,
        par_x=par_x,
        par_y=par_y,
        par_disa=par_disa,
        par_color=par_color,
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
    physicsname: str,
) -> None:
    save_dir_of_snap_path = save_dir_physics_path / Path("snap_shot")
    save_dir_of_snap_path.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(dpi=IN_PARAMS.snapshot_dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plot_colorbar(fig=fig, ax=ax, physics_name=physicsname)

    for snap_time_ms in snap_time_array_ms:
        make_snap_contour(
            fig=fig,
            ax=ax,
            snap_time_ms=snap_time_ms,
            save_dir_of_snap_path=save_dir_of_snap_path,
            physics_col_index=getattr(IN_PARAMS, f"{physicsname}_col_index"),
            physics_name=physicsname,
        )

    plt.close()

    return


def make_animation_contour(
    snap_time_array_ms: np.ndarray, save_dir_physics_path: Path, physicsname: str
) -> None:
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
            "60",  # TODO 与え方どうする？INPARAMか
            # f"{int(1000/IN_PARAMS.timestep_ms)}",  # TODO 与え方どうする？INPARAMか
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


IN_PARAMS = construct_input_parameters_dataclass()


def main() -> None:
    save_dir_path: Path = Path(__file__).parent / Path("plot_output_results")

    snap_time_array_ms: np.ndarray = np.arange(
        IN_PARAMS.snap_start_time_ms,
        IN_PARAMS.snap_end_time_ms + IN_PARAMS.timestep_ms,
        IN_PARAMS.timestep_ms,
    )

    print(
        f"animation range is: {snap_time_array_ms[0]/1000:.03f}[s] ~ {snap_time_array_ms[-1]/1000:.03f}[s]"
    )

    # 圧力コンターを出力
    cur_physicname = "pressure"
    save_dir_physics_path = save_dir_path / Path(cur_physicname)
    make_snap_contour_all_time(
        snap_time_array_ms=snap_time_array_ms,
        save_dir_physics_path=save_dir_physics_path,
        physicsname=cur_physicname,
    )
    make_animation_contour(
        snap_time_array_ms=snap_time_array_ms,
        save_dir_physics_path=save_dir_physics_path,
        physicsname=cur_physicname,
    )

    print("描画終了")

    return


if __name__ == "__main__":
    print(IN_PARAMS)
    main()
