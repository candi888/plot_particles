import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba

# 以下はイラレで編集可能なsvgを出力するために必要
matplotlib.use("Agg")
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


# TODO 型が合っているかのチェック含め，妥当なパラメータが設定されているかのチェック
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

    framerate: int

    pressure_label: str
    pressure_col_index: int
    pressure_min_value_contour: int
    pressure_max_value_contour: int
    pressure_cmap: str
    pressure_is_change_color_wall: bool
    pressure_wall_particle_color: str
    pressure_wall_particle_alpha: float
    pressure_is_change_color_dummy: bool
    pressure_dummy_particle_color: str
    pressure_dummy_particle_alpha: float
    pressure_is_plot_velocity_vector: bool

    vorticity_label: str
    vorticity_col_index: int
    vorticity_min_value_contour: int
    vorticity_max_value_contour: int
    vorticity_cmap: str
    vorticity_is_change_color_wall: bool
    vorticity_wall_particle_color: str
    vorticity_wall_particle_alpha: float
    vorticity_is_change_color_dummy: bool
    vorticity_dummy_particle_color: str
    vorticity_dummy_particle_alpha: float
    vorticity_is_plot_velocity_vector: bool

    crf_num: int = 35

    scaler_length_vector: float = 1
    scaler_width_vector: float = 1
    length_reference_vector: float = 1

    is_plot_contour_pressure: bool = False
    is_plot_contour_vorticity: bool = False


def construct_input_parameters_dataclass() -> Dataclass_Input_Parameters:
    with open(Path(__file__).parent / "INPUT_PARAMETERS.yaml", mode="r") as f:
        return Dataclass_Input_Parameters(**yaml.safe_load(f))


def get_mask_array_by_plot_region(snap_time_ms: int) -> np.ndarray:
    ori_data = np.loadtxt(
        Path(__file__).parent / Path(f"./OUTPUT/SNAP/XUD{snap_time_ms:05}.DAT"),
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


# TODO tmdの読み込みを別に分けるか．その上でmoveでグループ分けしてプロットもグループ分けするか
# TODO hako_keisokuのときとかも中身の値でグルーピングするか？
def change_facecolor_wall_particles(
    snap_time_ms: int, par_color: np.ndarray, physics_name: str, mask_array: np.ndarray
) -> None:
    masked_move_index = np.loadtxt(
        rf"./OUTPUT/SNAP/TMD{snap_time_ms:05}.DAT",
        dtype=np.int_,
        usecols=IN_PARAMS.TMD_move_col_index,
    )[mask_array]

    mask_by_wall = np.mod(masked_move_index, 3) == 1

    par_color[mask_by_wall] = to_rgba(
        c=getattr(IN_PARAMS, f"{physics_name}_wall_particle_color"),
        alpha=getattr(
            IN_PARAMS,
            f"{physics_name}_wall_particle_alpha",
        ),
    )

    return


def change_facecolor_dummy_particles(
    snap_time_ms: int, par_color: np.ndarray, physics_name: str, mask_array: np.ndarray
) -> None:
    masked_move_index = np.loadtxt(
        rf"./OUTPUT/SNAP/TMD{snap_time_ms:05}.DAT",
        dtype=np.int_,
        usecols=IN_PARAMS.TMD_move_col_index,
    )[mask_array]

    mask_by_wall = np.mod(masked_move_index, 3) == 2

    par_color[mask_by_wall] = to_rgba(
        c=getattr(IN_PARAMS, f"{physics_name}_dummy_particle_color"),
        alpha=getattr(
            IN_PARAMS,
            f"{physics_name}_dummy_particle_alpha",
        ),
    )

    return


# TODO  Thank you chatgpt
def data_unit_to_points_size(
    diameter_in_data_units: np.ndarray, fig: plt.Figure, axis: plt.Axes
) -> np.ndarray:
    """
    # TODO revise
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
    s = data_unit_to_points_size(diameter_in_data_units=par_disa, fig=fig, axis=ax)
    ax.scatter(par_x, par_y, s=s, c=par_color, linewidths=0)

    return


def get_norm_for_color_contour(physics_name: str) -> Normalize:
    return Normalize(
        vmin=getattr(IN_PARAMS, f"{physics_name}_min_value_contour"),
        vmax=getattr(IN_PARAMS, f"{physics_name}_max_value_contour"),
    )


def get_cmap_for_color_contour(physics_name: str) -> matplotlib.colors.Colormap:
    return matplotlib.colormaps.get_cmap(getattr(IN_PARAMS, f"{physics_name}_cmap"))


def get_facecolor_by_physics_contour(
    physics_name: str, par_physics: np.ndarray
) -> np.ndarray:
    cmap = get_cmap_for_color_contour(physics_name=physics_name)
    norm = get_norm_for_color_contour(physics_name=physics_name)
    par_color = cmap(norm(par_physics))

    return par_color


def plot_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    physics_name: str,
) -> None:
    cmap = get_cmap_for_color_contour(physics_name=physics_name)
    norm = get_norm_for_color_contour(physics_name=physics_name)
    mappable = ScalarMappable(cmap=cmap, norm=norm)

    assert norm.vmin is not None and norm.vmax is not None
    ticks = np.linspace(norm.vmin, norm.vmax, 5)

    fig.colorbar(
        mappable,
        ax=ax,
        ticks=ticks,
        shrink=0.6,
        orientation="horizontal",
        pad=0.12,
    ).set_label(f"{getattr(IN_PARAMS,f"{physics_name}_label")}")

    return


# TODO 矢印の見た目全般，移動壁への対応（その際のreference vectorへの対応），reference vectorがfloatのときのラベル
def plot_velocity_vector(
    ax: plt.Axes, snap_time_ms: int, mask_array: np.ndarray
) -> None:
    par_x, par_y, par_u, par_v = load_selected_par_data(
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=(
            IN_PARAMS.XUD_x_col_index,
            IN_PARAMS.XUD_y_col_index,
            IN_PARAMS.XUD_u_col_index,
            IN_PARAMS.XUD_v_col_index,
        ),
    )

    # scale_units="x"で軸の長さが基準
    # scale=10で，軸単位で0.1の長さが大きさ1のベクトルの長さに対応する
    original_scale = 10 / (IN_PARAMS.xlim_max - IN_PARAMS.xlim_min)
    scale = original_scale / IN_PARAMS.scaler_length_vector
    width = original_scale / 5000 * IN_PARAMS.scaler_width_vector
    q = ax.quiver(par_x, par_y, par_u, par_v, scale=scale, scale_units="x", width=width)
    ax.quiverkey(
        Q=q,
        X=0.8,
        Y=1.1,
        U=IN_PARAMS.length_reference_vector,
        label=rf"{IN_PARAMS.length_reference_vector} m/s",
        labelpos="E",
    )

    return


def set_ax_ticks(ax: plt.Axes) -> None:
    ax.set_xlim(IN_PARAMS.xlim_min, IN_PARAMS.xlim_max)
    ax.set_ylim(IN_PARAMS.ylim_min, IN_PARAMS.ylim_max)

    return


def set_ax_labels(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$x \mathrm{(m)}$")
    ax.set_ylabel(r"$y \mathrm{(m)}$")
    return


def set_ax_title(ax: plt.Axes, snap_time_ms: int) -> None:
    ax.set_title(rf"$t=$ {snap_time_ms/1000:.03f}s", y=1.1, loc="left")

    return


# TODO 色だけでグループ分けするか，色の配列作って最後にgroupby
def make_snap_contour(
    fig: plt.Figure,
    ax: plt.Axes,
    snap_time_ms: int,
    save_dir_snap_path: Path,
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
    par_color = get_facecolor_by_physics_contour(
        physics_name=physics_name, par_physics=par_physics
    )

    if getattr(IN_PARAMS, f"{physics_name}_is_change_color_wall"):
        change_facecolor_wall_particles(
            snap_time_ms=snap_time_ms,
            par_color=par_color,
            physics_name=physics_name,
            mask_array=mask_array,
        )
    if getattr(IN_PARAMS, f"{physics_name}_is_change_color_dummy"):
        change_facecolor_dummy_particles(
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

    if getattr(IN_PARAMS, f"{physics_name}_is_plot_velocity_vector"):
        plot_velocity_vector(ax=ax, snap_time_ms=snap_time_ms, mask_array=mask_array)

    fig.savefig(
        save_dir_snap_path / Path(f"snap{snap_time_ms:05}_{physics_name}.jpeg"),
    )

    plt.cla()
    print(f"{snap_time_ms/1000:.03f} s contour:{physics_name} plot finished")

    return


def make_snap_contour_all_time(
    snap_time_array_ms: np.ndarray,
    save_dir_physics_path: Path,
    physics_name: str,
) -> None:
    save_dir_snap_path = save_dir_physics_path / Path("snap_shot")
    save_dir_snap_path.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(dpi=IN_PARAMS.snapshot_dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plot_colorbar(fig=fig, ax=ax, physics_name=physics_name)

    for snap_time_ms in snap_time_array_ms:
        make_snap_contour(
            fig=fig,
            ax=ax,
            snap_time_ms=snap_time_ms,
            save_dir_snap_path=save_dir_snap_path,
            physics_col_index=getattr(IN_PARAMS, f"{physics_name}_col_index"),
            physics_name=physics_name,
        )

    plt.close()
    return


# TODO subporocessのとこリファクタリング
def make_animation_contour(
    snap_time_array_ms: np.ndarray, save_dir_physics_path: Path, physics_name: str
) -> None:
    save_dir_animation_path = save_dir_physics_path / Path("animation")
    save_dir_animation_path.mkdir(exist_ok=True)

    for_ffmpeg = [f"file 'snap{i:05}_{physics_name}.jpeg'" for i in snap_time_array_ms]
    save_file_forffmpeg_path = save_dir_physics_path / Path("snap_shot/for_ffmpeg.txt")
    with open(save_file_forffmpeg_path, mode="w") as f:
        for i in for_ffmpeg:
            f.write(f"{i}\n")

    cmd_list1 = [
        "ffmpeg",
        "-y",
        "-r",
        f"{IN_PARAMS.framerate}",
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
    ]
    cmd_list2 = ["-pix_fmt", "yuv420p"]

    cur_save_file_name = f"{physics_name}.mp4"
    subprocess.run(
        cmd_list1
        + cmd_list2
        + [str(save_dir_animation_path / Path(cur_save_file_name))],
        cwd=str(save_dir_physics_path / Path("snap_shot")),
    )

    # 以下は低画質用
    cur_save_file_name = f"{physics_name}_lowquality.mp4"
    subprocess.run(
        cmd_list1
        + ["-crf", f"{IN_PARAMS.crf_num}"]  # ここで動画の品質を調整
        + cmd_list2
        + [str(save_dir_animation_path / Path(cur_save_file_name))],
        cwd=str(save_dir_physics_path / Path("snap_shot")),
    )

    print(f"contour:{physics_name} animation finished")

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

    # コンター図を作成
    for cur_physics_name in ["pressure", "vorticity"]:
        if getattr(IN_PARAMS, f"is_plot_contour_{cur_physics_name}"):
            save_dir_physics_path = save_dir_path / Path(cur_physics_name)
            make_snap_contour_all_time(
                snap_time_array_ms=snap_time_array_ms,
                save_dir_physics_path=save_dir_physics_path,
                physics_name=cur_physics_name,
            )
            make_animation_contour(
                snap_time_array_ms=snap_time_array_ms,
                save_dir_physics_path=save_dir_physics_path,
                physics_name=cur_physics_name,
            )

    print("描画終了")

    return


if __name__ == "__main__":
    main()
