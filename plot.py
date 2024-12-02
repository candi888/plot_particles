import dataclasses
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba
from numpy.typing import NDArray

mplstyle.use("fast")

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
# plt.rcParams["xtick.top"] = True  # x軸の上部目盛り
# plt.rcParams["ytick.right"] = True  # y軸の右部目盛り
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

    # * 渦動粘性係数コンター作成関係
    eddy_viscos_label: str
    eddy_viscos_col_index: int
    eddy_viscos_min_value_contour: float
    eddy_viscos_max_value_contour: float
    eddy_viscos_cmap: str
    eddy_viscos_is_plot_velocity_vector: bool

    #!　以下，デフォルト値あり

    # * svg出力用．これをTrueにした場合，ここで指定した時刻のsvgのみを作成する．jpegの画像やアニメーションの作成は行わないので注意．
    svg_flag: bool = False
    svg_snap_time_ms: int | None = None

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

    # * speedコンター作成関係
    eddy_viscos_wall_particle_color: str | None = None
    eddy_viscos_wall_particle_alpha: float = 1.0
    eddy_viscos_dummy_particle_color: str | None = None
    eddy_viscos_dummy_particle_alpha: float = 1.0
    eddy_viscos_movewall_particle_color: str | None = None
    eddy_viscos_movewall_particle_alpha: float = 1.0
    eddy_viscos_movedummy_particle_color: str | None = None
    eddy_viscos_movedummy_particle_alpha: float = 1.0

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

        # svg関連の処理
        if self.svg_flag and self.svg_snap_time_ms is None:
            raise ValueError(
                "svg_snap time_msの値が設定されていません．svgを作成したい時刻[ms]を設定してください"
            )

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
    margin = np.max(original_data[:, 2]) * 2  # 最大粒径の2倍分marginを設定

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


# TODO revise
def postprocess_svg_setclippath(
    svg_file_path: Path, particle_group_id_prefix: str, vector_group_id_prefix: str
) -> None:
    """
    指定されたSVGファイルに対して，'{particle_group_id_prefix}'または'{vector_group_id_prefix}'という文字列を含むIDを持つ要素に
    クリッピングマスクを適用し，修正したファイルを上書き保存する．

    Args:
        svg_file_path (Path): 修正対象のSVGファイルのパス
        particle_group_id_prefix (str): 粒子のグループID
        vector_group_id_prefix (str): 流速ベクトルのグループID
    """
    # SVGファイルをパースしてツリー構造を取得
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    # # クリッピングパスが定義されていない場合は、<defs>セクションに追加
    defs = root.find("{http://www.w3.org/2000/svg}defs")
    if defs is None:
        defs = ET.Element("{http://www.w3.org/2000/svg}defs")
        root.insert(0, defs)

    # クリッピングパスを定義
    clip_path = ET.Element("{http://www.w3.org/2000/svg}clipPath", {"id": "clip-axis"})
    # TODO 一般性を
    clip_rect = ET.Element(
        "{http://www.w3.org/2000/svg}rect",
        {
            "x": "41.201563",  # 軸の開始位置
            "y": "21.9325",  # 軸の下限
            "width": "460.8",  # 軸の幅
            "height": "25",  # 軸の高さ
        },
    )
    clip_path.append(clip_rect)
    defs.append(clip_path)

    # '{particle_group_id_prefix}'または'{vector_group_id_prefix}'という文字列を含むIDを持つグループに対してクリッピングマスクを適用
    for group in root.findall(".//{http://www.w3.org/2000/svg}g"):
        group_id = group.attrib.get("id", "")
        if (particle_group_id_prefix in group_id) or (
            vector_group_id_prefix in group_id
        ):
            group.set("clip-path", "url(#clip-axis)")

    # 修正内容を元のファイルに上書き保存
    tree.write(svg_file_path, encoding="utf-8", xml_declaration=True)

    return


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
    group_id_prefix: str,
    group_index: int,
) -> None:
    s = data_unit_to_points_size(diameter_in_data_units=par_disa, fig=fig, axis=ax)
    ax.scatter(
        par_x,
        par_y,
        s=s,
        c=par_color,
        linewidths=0,
        gid=f"{group_id_prefix}{group_index}",
        # clip_on=not IN_PARAMS.svg_flag,
    )

    return


def get_norm_for_color_contour(physics_name: str) -> Normalize:
    return Normalize(
        vmin=getattr(IN_PARAMS, f"{physics_name}_min_value_contour"),
        vmax=getattr(IN_PARAMS, f"{physics_name}_max_value_contour"),
    )


def get_cmap_for_color_contour(physics_name: str) -> Colormap:
    cmap_name = getattr(IN_PARAMS, f"{physics_name}_cmap")

    try:
        return (
            LinearSegmentedColormap.from_list(
                "custom", ["blue", "cyan", "lime", "yellow", "red"], N=512
            )
            if cmap_name == "small rainbow"
            else mpl.colormaps.get_cmap(cmap_name)
        )
    except ValueError:
        raise ValueError(
            f'{physics_name}_cmap で設定されているcolormap名は存在しません．\nhttps://matplotlib.org/stable/users/explain/colors/colormaps.html に載っているcolormap名，もしくは"small rainbow"を設定してください．'
        )


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

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        ticks=ticks,
        shrink=0.28,
        orientation="horizontal",
        # pad=0.12,
        location="top",
        anchor=(0.5, 0.0),
        ticklocation="bottom",
    )
    cbar.ax.xaxis.set_ticks_position("bottom")
    # cbar.ax.xaxis.set_label_position("bottom") # スナップの方に貫通するので注意
    cbar.set_label(f"{getattr(IN_PARAMS,f"{physics_name}_label")}")
    cbar.minorticks_off()

    return


def plot_velocity_vector(
    fig: plt.Figure,
    ax: plt.Axes,
    snap_time_ms: int,
    is_plot_reference_vector: bool,
    mask_array: NDArray[np.bool_],
    group_id_prefix: str,
    group_index: int,
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
    q = ax.quiver(
        par_x,
        par_y,
        par_u,
        par_v,
        scale=scale,
        scale_units="x",
        width=width,
        # width=0 if IN_PARAMS.svg_flag else width,
        # headwidth=0 if IN_PARAMS.svg_flag else 3.0,
        gid=f"{group_id_prefix}{group_index}",
        # clip_on=not IN_PARAMS.svg_flag,
    )

    if is_plot_reference_vector:
        title = ax.title
        # タイトルのバウンディングボックスを取得
        renderer = fig.canvas.get_renderer()
        bbox_title = title.get_window_extent(renderer=renderer)

        y_center_display = bbox_title.y0

        # 表示座標系から軸座標系への変換
        y_center_axes = ax.transAxes.inverted().transform((0, y_center_display))[1]

        ax.quiverkey(
            Q=q,
            X=0.9,
            Y=y_center_axes,
            U=IN_PARAMS.length_reference_vector,
            label=f"Velocity $\\mathbfit{{u}}$\n{IN_PARAMS.length_reference_vector} m/s",
            labelpos="N",
        )

    return


def set_ax_ticks(ax: plt.Axes) -> None:
    ax.set_xlim(IN_PARAMS.xlim_min, IN_PARAMS.xlim_max)
    ax.set_ylim(IN_PARAMS.ylim_min, IN_PARAMS.ylim_max)

    return


def set_ax_labels(ax: plt.Axes) -> None:
    # mathtextを使用するとIllustratorでsvgを読み込んだ時にテキストの編集がしづらくなってしまうので以下のように対応
    xlabel = "x (m)" if IN_PARAMS.svg_flag else r"$x \mathrm{(m)}$"
    ylabel = "y (m)" if IN_PARAMS.svg_flag else r"$y \mathrm{(m)}$"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return


def set_ax_title(ax: plt.Axes, snap_time_ms: int) -> None:
    time_text = f"{snap_time_ms/1000:.03f}s"

    # mathtextを使用するとIllustratorでsvgを読み込んだ時にテキストの編集がしづらくなってしまうので以下のように対応
    title_text = f"t= {time_text}" if IN_PARAMS.svg_flag else rf"$t=$ {time_text}"

    ax.set_title(
        title_text,
        pad=7,
        loc="left",
    )

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

    particle_group_id_prefix = "particle_group"
    vector_group_id_prefix = "vector_group"

    # TODO if physics_name == "splash" else get_splash...

    is_plot_reference_vector = True
    for group_index in np.unique(par_group_index)[::-1]:
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
            group_id_prefix=particle_group_id_prefix,
            group_index=group_index,
        )

        if getattr(IN_PARAMS, f"{physics_name}_is_plot_velocity_vector") and (
            group_index not in {1, 2}
        ):
            plot_velocity_vector(
                fig=fig,
                ax=ax,
                snap_time_ms=snap_time_ms,
                mask_array=mask_array,
                is_plot_reference_vector=is_plot_reference_vector,
                mask_array_by_group=mask_array_by_group,
                group_id_prefix=vector_group_id_prefix,
                group_index=group_index,
            )
            # 初回のみreference vectorをプロット
            is_plot_reference_vector = False

    # 以下，画像の保存処理
    save_file_name_without_extension = f"snap{snap_time_ms:05}_{physics_name}"
    extension = "svg" if IN_PARAMS.svg_flag else "jpeg"
    save_file_path = save_dir_snap_path / Path(
        f"{save_file_name_without_extension}.{extension}"
    )
    fig.savefig(save_file_path)

    # TODO revise
    # if IN_PARAMS.svg_flag:
    #     postprocess_svg_setclippath(
    #         svg_file_path=save_file_path,
    #         particle_group_id_prefix=particle_group_id_prefix,
    #         vector_group_id_prefix=vector_group_id_prefix,
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


def make_snap_physics_contour_svg(
    save_dir_path: Path, physics_name: str, is_move_or_splash: bool
) -> None:
    save_dir_path.mkdir(exist_ok=True, parents=True)

    snap_time_ms = IN_PARAMS.svg_snap_time_ms
    assert (
        snap_time_ms is not None
    )  # __post_init__でチェック済みだががwarningをなくすために記述

    fig = plt.figure(dpi=IN_PARAMS.snapshot_dpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if not is_move_or_splash:
        plot_colorbar(fig=fig, ax=ax, physics_name=physics_name)

    try:
        print(
            f"時刻{snap_time_ms/1000:.03f} sのsvgを作成します．jpegのsnapやアニメーションはここでは作成されません．"
        )
        make_snap_physics_contour(
            fig=fig,
            ax=ax,
            snap_time_ms=snap_time_ms,
            save_dir_snap_path=save_dir_path,
            physics_name=physics_name,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{snap_time_ms/1000:.03f} s時点の計算データがありません．\n"
        )

    print(f"{snap_time_ms/1000:.03f} s contour:{physics_name} make svg finished\n")
    plt.close()

    return


IN_PARAMS = construct_input_parameters_dataclass()


def main() -> None:
    # 指定したスナップ作成時間のデータが存在しないときは処理を止める
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

        # svg出力
        if IN_PARAMS.svg_flag:
            make_snap_physics_contour_svg(
                save_dir_path=save_dir_path,
                physics_name=cur_physics_name,
                is_move_or_splash=is_move_or_splash,
            )
            continue

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

    print("all finished")

    return


if __name__ == "__main__":
    main()
#
