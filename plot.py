import dataclasses
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
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


@dataclasses.dataclass(frozen=True)
class DataclassInputParameters:
    # * スナップやアニメーションを保存するディレクトリ名
    save_dir_name: str

    # * プロットの順序．上から順番にプロットを行う．いらないものはコメントアウトすればプロットされない．
    plot_order_list: (
        list | None
    )  # リストの要素がゼロのときはNoneになる．List[str]であるが，後の__post_init__でチェック

    # * 物理量が素データのどの軸か．0-index
    XUD_x_col_index: int
    XUD_y_col_index: int
    XUD_u_col_index: int
    XUD_v_col_index: int
    XUD_disa_col_index: int
    TMD_move_col_index: int

    # * プロット関連
    timestep_ms: int
    snap_start_time_ms: int
    snap_end_time_ms: int
    xlim_min: float
    xlim_max: float
    ylim_min: float
    ylim_max: float
    snapshot_dpi: int

    framerate: int

    # * 圧力コンター作成関係
    pressure_label: str
    pressure_col_index: int
    pressure_min_value_contour: int
    pressure_max_value_contour: int
    pressure_cmap: str
    pressure_is_plot_velocity_vector: bool

    # * 渦度コンター作成関係
    vorticity_label: str
    vorticity_col_index: int
    vorticity_min_value_contour: int
    vorticity_max_value_contour: int
    vorticity_cmap: str
    vorticity_is_plot_velocity_vector: bool

    # * Q_signedコンター作成関係
    Q_signed_label: str
    Q_signed_col_index: int
    Q_signed_min_value_contour: int
    Q_signed_max_value_contour: int
    Q_signed_cmap: str
    Q_signed_is_plot_velocity_vector: bool

    # * densityコンター作成関係
    density_label: str
    density_col_index: int
    density_min_value_contour: float
    density_max_value_contour: float
    density_cmap: str
    density_is_plot_velocity_vector: bool

    # * div_Uコンター作成関係
    div_U_label: str
    div_U_col_index: int
    div_U_min_value_contour: float
    div_U_max_value_contour: float
    div_U_cmap: str
    div_U_is_plot_velocity_vector: bool

    # * speedコンター作成関係
    speed_label: str
    speed_col_index: int
    speed_min_value_contour: float
    speed_max_value_contour: float
    speed_cmap: str
    speed_is_plot_velocity_vector: bool

    #!　以下，デフォルト値あり

    # * 圧力コンター作成関係
    pressure_wall_particle_color: str | None = None
    pressure_wall_particle_alpha: float = 1.0
    pressure_dummy_particle_color: str | None = None
    pressure_dummy_particle_alpha: float = 1.0
    pressure_movewall_particle_color: str | None = None
    pressure_movewall_particle_alpha: float = 1.0
    pressure_movedummy_particle_color: str | None = None
    pressure_movedummy_particle_alpha: float = 1.0

    # * 渦度コンター作成関係
    vorticity_wall_particle_color: str | None = None
    vorticity_wall_particle_alpha: float = 1.0
    vorticity_dummy_particle_color: str | None = None
    vorticity_dummy_particle_alpha: float = 1.0
    vorticity_movewall_particle_color: str | None = None
    vorticity_movewall_particle_alpha: float = 1.0
    vorticity_movedummy_particle_color: str | None = None
    vorticity_movedummy_particle_alpha: float = 1.0

    # * Q_signedコンター作成関係
    Q_signed_wall_particle_color: str | None = None
    Q_signed_wall_particle_alpha: float = 1.0
    Q_signed_dummy_particle_color: str | None = None
    Q_signed_dummy_particle_alpha: float = 1.0
    Q_signed_movewall_particle_color: str | None = None
    Q_signed_movewall_particle_alpha: float = 1.0
    Q_signed_movedummy_particle_color: str | None = None
    Q_signed_movedummy_particle_alpha: float = 1.0

    # * densityコンター作成関係
    density_wall_particle_color: str | None = None
    density_wall_particle_alpha: float = 1.0
    density_dummy_particle_color: str | None = None
    density_dummy_particle_alpha: float = 1.0
    density_movewall_particle_color: str | None = None
    density_movewall_particle_alpha: float = 1.0
    density_movedummy_particle_color: str | None = None
    density_movedummy_particle_alpha: float = 1.0

    # * div_Uコンター作成関係
    div_U_wall_particle_color: str | None = None
    div_U_wall_particle_alpha: float = 1.0
    div_U_dummy_particle_color: str | None = None
    div_U_dummy_particle_alpha: float = 1.0
    div_U_movewall_particle_color: str | None = None
    div_U_movewall_particle_alpha: float = 1.0
    div_U_movedummy_particle_color: str | None = None
    div_U_movedummy_particle_alpha: float = 1.0

    # * speedコンター作成関係
    speed_wall_particle_color: str | None = None
    speed_wall_particle_alpha: float = 1.0
    speed_dummy_particle_color: str | None = None
    speed_dummy_particle_alpha: float = 1.0
    speed_movewall_particle_color: str | None = None
    speed_movewall_particle_alpha: float = 1.0
    speed_movedummy_particle_color: str | None = None
    speed_movedummy_particle_alpha: float = 1.0

    # * move関連
    move_is_plot_velocity_vector: bool = True
    move_water_particle_color: str = "#00FFFF"
    move_water_particle_alpha: float = 1.0
    move_wall_particle_color: str = "#BC8F8F"
    move_wall_particle_alpha: float = 1.0
    move_dummy_particle_color: str = "#A52A2A"
    move_dummy_particle_alpha: float = 1.0
    move_movewall_particle_color: str = "#ff1493"
    move_movewall_particle_alpha: float = 1.0
    move_movedummy_particle_color: str = "#ff69b4"
    move_movedummy_particle_alpha: float = 1.0

    crf_num: int = 35

    scaler_length_vector: float = 1.0
    scaler_width_vector: float = 1.0
    length_reference_vector: int = 1

    def __post_init__(self) -> None:
        class_dict = dataclasses.asdict(self)
        # * 1. 型チェック
        # self.__annotations__は {引数の名前:指定する型}
        for class_arg_name, class_arg_expected_type in self.__annotations__.items():
            if not isinstance(class_dict[class_arg_name], class_arg_expected_type):
                raise ValueError(
                    f"{class_arg_name}の型が一致しません．\n{class_arg_name}の型は現在は{type(class_dict[class_arg_name])}ですが，{class_arg_expected_type}である必要があります．"
                )

        # plot_order_listの処理
        if self.plot_order_list is not None:
            if not all([isinstance(name, str) for name in self.plot_order_list]):
                raise ValueError(
                    f"plot_order_listの要素の型が一致しません．\nplot_order_listの中身の型は全て{str}である必要があります．"
                )
        print("1. 型チェック OK")

        # * 2.諸々のエラー処理

        print("IN_PARAMS construct OK\n")
        return


def construct_input_parameters_dataclass() -> DataclassInputParameters:
    # print("Please input file name(with extension):")
    input_yaml_name = sys.argv[1]

    with open(
        Path(__file__).parent / "input_yaml" / f"{input_yaml_name}",
        mode="r",
        encoding="utf-8",
    ) as f:
        print(f"plot execute by {input_yaml_name}\n")
        return DataclassInputParameters(**yaml.safe_load(f))


def get_mask_array_by_plot_region(snap_time_ms: int) -> NDArray[np.bool_]:
    original_data = np.loadtxt(
        Path(__file__).parent / "OUTPUT" / "SNAP" / f"XUD{snap_time_ms:05}.DAT",
        usecols=(
            IN_PARAMS.XUD_x_col_index,
            IN_PARAMS.XUD_y_col_index,
            IN_PARAMS.XUD_disa_col_index,
        ),
        dtype=np.float64,
    )

    x = original_data[:, 0]
    y = original_data[:, 1]
    margin = np.max(original_data[:, 2]) * 3  # 最大粒径の３倍分marginを設定

    # 範囲条件を設定
    mask_array = (
        (x >= IN_PARAMS.xlim_min - margin)
        & (x <= IN_PARAMS.xlim_max + margin)
        & (y >= IN_PARAMS.ylim_min - margin)
        & (y <= IN_PARAMS.ylim_max + margin)
    )

    return mask_array


def load_par_data_masked_by_plot_region(
    snap_time_ms: int,
    usecols: tuple[int, ...],
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    original_data = np.loadtxt(
        Path(__file__).parent / "OUTPUT" / "SNAP" / f"XUD{snap_time_ms:05}.DAT",
        usecols=usecols,
        dtype=np.float64,
    )

    masked_data = original_data[mask_array]

    if mask_array_by_group is not None:
        masked_data = masked_data[mask_array_by_group]

    return masked_data.T


def get_move_index_array(
    snap_time_ms: int, mask_array: NDArray[np.bool_]
) -> NDArray[np.int8]:
    masked_move_index = np.loadtxt(
        Path(__file__).parent / "OUTPUT" / "SNAP" / f"TMD{snap_time_ms:05}.DAT",
        dtype=np.int8,
        usecols=IN_PARAMS.TMD_move_col_index,
    )[mask_array]

    return masked_move_index


def data_unit_to_points_size(
    diameter_in_data_units: NDArray[np.float64], fig: plt.Figure, axis: plt.Axes
) -> NDArray[np.float64]:
    """
    データ単位で指定した直径を、matplotlib の scatter プロットで使用する s パラメータ（ポイントの面積）に変換します。

    Parameters
    ----------
    diameter_in_data_units : NDArray[np.float64]
        データ単位での直径の配列。
    axis : matplotlib.axes.Axes
        対象の Axes オブジェクト。

    Returns
    -------
    NDArray[np.float64]
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
    par_x: NDArray[np.float64],
    par_y: NDArray[np.float64],
    par_disa: NDArray[np.float64],
    par_color: NDArray[np.float64],
) -> None:
    s = data_unit_to_points_size(diameter_in_data_units=par_disa, fig=fig, axis=ax)
    ax.scatter(par_x, par_y, s=s, c=par_color, linewidths=0)

    return


def get_norm_for_color_contour(physics_name: str) -> Normalize:
    return Normalize(
        vmin=getattr(IN_PARAMS, f"{physics_name}_min_value_contour"),
        vmax=getattr(IN_PARAMS, f"{physics_name}_max_value_contour"),
    )


def get_cmap_for_color_contour(physics_name: str) -> mpl.colors.Colormap:
    return mpl.colormaps.get_cmap(getattr(IN_PARAMS, f"{physics_name}_cmap"))


def get_facecolor_by_physics_contour(
    physics_name: str,
    snap_time_ms: int,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    cmap = get_cmap_for_color_contour(physics_name=physics_name)
    norm = get_norm_for_color_contour(physics_name=physics_name)

    par_physics = load_par_data_masked_by_plot_region(
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=getattr(IN_PARAMS, f"{physics_name}_col_index"),
        mask_array_by_group=mask_array_by_group,
    )

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


def plot_velocity_vector(
    ax: plt.Axes,
    snap_time_ms: int,
    is_plot_reference_vector: bool,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> None:
    par_x, par_y, par_u, par_v = load_par_data_masked_by_plot_region(
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=(
            IN_PARAMS.XUD_x_col_index,
            IN_PARAMS.XUD_y_col_index,
            IN_PARAMS.XUD_u_col_index,
            IN_PARAMS.XUD_v_col_index,
        ),
        mask_array_by_group=mask_array_by_group,
    )

    # scale_units="x"で軸の長さが基準
    # scale=10で，軸単位で0.1の長さが大きさ1のベクトルの長さに対応する
    original_scale = 10 / (IN_PARAMS.xlim_max - IN_PARAMS.xlim_min)
    scale = original_scale / IN_PARAMS.scaler_length_vector
    width = original_scale / 5000 * IN_PARAMS.scaler_width_vector
    q = ax.quiver(par_x, par_y, par_u, par_v, scale=scale, scale_units="x", width=width)

    if is_plot_reference_vector:
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
    ax.set_title(rf"$t=$ {snap_time_ms/1000:.03f}s", pad=10, loc="left")
    return


def change_facecolor_by_move(
    par_color_masked_by_move: NDArray[np.float64],
    physics_name: str,
    move_label: str,
) -> NDArray[np.float64]:
    input_color = getattr(IN_PARAMS, f"{physics_name}_{move_label}_particle_color")
    input_alpha = getattr(IN_PARAMS, f"{physics_name}_{move_label}_particle_alpha")

    if input_color is None:
        par_color_masked_by_move[:, 3] = input_alpha  # ４列目がrgbaのa
    else:
        par_color_masked_by_move = np.full(
            par_color_masked_by_move.shape, to_rgba(c=input_color, alpha=input_alpha)
        )

    return par_color_masked_by_move


def get_facecolor_array_for_move_or_physics_contour(
    is_move: bool,
    move_index: int,
    physics_name: str,
    snap_time_ms: int,
    num_par_cur_group: int,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_],
) -> NDArray[np.float64]:
    move_to_movelabel = {
        0: "water",
        1: "wall",
        2: "dummy",
        4: "movewall",
        5: "movedummy",
    }

    if is_move:
        par_color = np.full((num_par_cur_group, 4), -1, dtype=np.float64)
    else:
        par_color = get_facecolor_by_physics_contour(
            physics_name=physics_name,
            snap_time_ms=snap_time_ms,
            mask_array=mask_array,
            mask_array_by_group=mask_array_by_group,
        )

    # move以外のコンター図 かつ 水粒子　-> 色を変えずそのまま
    if (not is_move) and move_index % 3 == 0:
        return par_color

    return change_facecolor_by_move(
        par_color_masked_by_move=par_color,
        physics_name=physics_name,
        move_label=move_to_movelabel[move_index],
    )


def make_snap_physics_contour(
    fig: plt.Figure,
    ax: plt.Axes,
    snap_time_ms: int,
    save_dir_snap_path: Path,
    physics_name: str,
) -> None:
    set_ax_ticks(ax=ax)
    set_ax_labels(ax=ax)
    set_ax_title(ax=ax, snap_time_ms=snap_time_ms)

    # TODO キャンバス更新
    if snap_time_ms == IN_PARAMS.snap_start_time_ms:
        fig.canvas.draw()

    mask_array = get_mask_array_by_plot_region(snap_time_ms=snap_time_ms)

    par_group_index = get_move_index_array(
        snap_time_ms=snap_time_ms, mask_array=mask_array
    )
    # TODO if physics_name == "splash" else get_splash...

    # TODO リファクタリング
    # TODO sort()[::-1]は水粒子を最後にプロットして一番上に置くため
    for iter, group_index in enumerate(np.unique(par_group_index)[::-1]):
        mask_array_by_group: NDArray[np.bool_] = par_group_index == group_index

        par_x, par_y, par_disa = load_par_data_masked_by_plot_region(
            snap_time_ms=snap_time_ms,
            mask_array=mask_array,
            usecols=(
                IN_PARAMS.XUD_x_col_index,
                IN_PARAMS.XUD_y_col_index,
                IN_PARAMS.XUD_disa_col_index,
            ),
            mask_array_by_group=mask_array_by_group,
        )

        if physics_name == "splash":
            pass  # TODO
        else:
            is_move: bool = physics_name == "move"
            par_color = get_facecolor_array_for_move_or_physics_contour(
                is_move=is_move,
                move_index=group_index,
                physics_name=physics_name,
                snap_time_ms=snap_time_ms,
                num_par_cur_group=par_x.shape[0],
                mask_array=mask_array,
                mask_array_by_group=mask_array_by_group,
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
            plot_velocity_vector(
                ax=ax,
                snap_time_ms=snap_time_ms,
                mask_array=mask_array,
                is_plot_reference_vector=iter
                == 0,  # 最初の一回のみreference vectorをプロットする
                mask_array_by_group=mask_array_by_group,
            )

    fig.savefig(save_dir_snap_path / f"snap{snap_time_ms:05}_{physics_name}.jpeg")
    # if physics_name == "move" and snap_time_ms == 50:
    #     fig.savefig(
    #         Path(__file__).parent / Path(f"snap{snap_time_ms:05}_{physics_name}.svg"),
    #     )

    plt.cla()

    return


def make_snap_physics_contour_all_snap_time(
    snap_time_array_ms: NDArray[np.int64],
    save_dir_physics_path: Path,
    physics_name: str,
    is_move_or_splash: bool,
) -> None:
    save_dir_snap_path = save_dir_physics_path / "snap_shot"
    save_dir_snap_path.mkdir(exist_ok=True, parents=True)

    fig = plt.figure(dpi=IN_PARAMS.snapshot_dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if not is_move_or_splash:
        plot_colorbar(fig=fig, ax=ax, physics_name=physics_name)

    for snap_time_ms in snap_time_array_ms:
        try:
            make_snap_physics_contour(
                fig=fig,
                ax=ax,
                snap_time_ms=snap_time_ms,
                save_dir_snap_path=save_dir_snap_path,
                physics_name=physics_name,
            )
            print(f"{snap_time_ms/1000:.03f} s contour:{physics_name} plot finished")
        except FileNotFoundError:
            print(
                f"{snap_time_ms/1000:.03f} s時点の計算データがありません．スナップショットの作成を終了します．\n"
            )
            break

    print(f"{snap_time_ms/1000:.03f} s contour:{physics_name} all make snap finished\n")
    plt.close()
    return


# TODO subporocessのとこリファクタリング
def make_animation_from_snap(
    snap_time_array_ms: NDArray[np.int64], save_dir_sub_path: Path, physics_name: str
) -> None:
    save_dir_animation_path = save_dir_sub_path / "animation"
    save_dir_animation_path.mkdir(exist_ok=True)

    # 連番でない画像の読み込みに対応させるための準備
    for_ffmpeg = []
    for snap_time_ms in snap_time_array_ms:
        cur_snap_path = (
            save_dir_sub_path
            / "snap_shot"
            / f"snap{snap_time_ms:05}_{physics_name}.jpeg"
        )
        # アニメーション作成で使うsnapが存在するかの確認
        if not cur_snap_path.exists():
            break
        for_ffmpeg.append(f"file 'snap{snap_time_ms:05}_{physics_name}.jpeg'")

    if for_ffmpeg == []:
        print(
            "アニメーション作成対象のスナップショットがありません．アニメーション作成を終了します.\n"
        )
        return

    print(f"animation range: {for_ffmpeg[0]} ~ {for_ffmpeg[-1]}")

    # 一度ファイルに書き込んでからffmpegで読み取る．あとでこのファイルは削除
    save_file_forffmpeg_path = save_dir_sub_path / "snap_shot" / "tmp_for_ffmpeg.txt"
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
        "-loglevel",
        "warning",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-vcodec",
        "libx264",
    ]
    cmd_list2 = ["-pix_fmt", "yuv420p"]

    cur_save_file_name = f"{physics_name}.mp4"
    subprocess.run(
        cmd_list1 + cmd_list2 + [str(save_dir_animation_path / cur_save_file_name)],
        cwd=str(save_dir_sub_path / "snap_shot"),
    )

    # 以下は低画質用
    cur_save_file_name = f"{physics_name}_lowquality.mp4"
    subprocess.run(
        cmd_list1
        + ["-crf", f"{IN_PARAMS.crf_num}"]  # ここで動画の品質を調整
        + cmd_list2
        + [str(save_dir_animation_path / cur_save_file_name)],
        cwd=str(save_dir_sub_path / "snap_shot"),
    )

    # tmp_for_ffmpeg.txtを削除
    save_file_forffmpeg_path.unlink()

    print(f"{physics_name} animation finished\n")

    return


IN_PARAMS = construct_input_parameters_dataclass()


def main() -> None:
    if IN_PARAMS.plot_order_list is None:
        exit()

    # スナップやアニメーションを保存するディレクトリ名
    save_dir_path: Path = Path(__file__).parent / IN_PARAMS.save_dir_name

    # スナップショットを出力する時間[ms]のarray
    snap_time_array_ms: NDArray[np.int64] = np.arange(
        IN_PARAMS.snap_start_time_ms,
        IN_PARAMS.snap_end_time_ms + IN_PARAMS.timestep_ms,
        IN_PARAMS.timestep_ms,
    )

    # * コンター図を作成
    for cur_physics_name in IN_PARAMS.plot_order_list:
        save_dir_physics_path = save_dir_path / cur_physics_name
        is_move_or_splash: bool = (
            cur_physics_name == "move" or cur_physics_name == "splash"
        )

        # TODO ↓いつかリファクタリングしたい
        # * moveとsplashのみは物理量によるコンター図にはしない
        make_snap_physics_contour_all_snap_time(
            snap_time_array_ms=snap_time_array_ms,
            save_dir_physics_path=save_dir_physics_path,
            physics_name=cur_physics_name,
            is_move_or_splash=is_move_or_splash,
        )
        make_animation_from_snap(
            snap_time_array_ms=snap_time_array_ms,
            save_dir_sub_path=save_dir_physics_path,
            physics_name=cur_physics_name,
        )
        print(f"contour:{cur_physics_name} finished\n")

    print("all finished\n")

    return


if __name__ == "__main__":
    main()
