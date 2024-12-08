import dataclasses
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.figure import Figure
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class DataclassInputParameters:
    # * スナップやアニメーションを保存するディレクトリ名
    save_dir_name: str

    # * contourプロットの順序．上から順番にプロットを行う．
    plot_order_list_contour: list  # List[str]であるが，後の__post_init__でチェック

    # * groupプロットの順序．上から順番にプロットを行う．
    plot_order_list_group: list  # List[str]であるが，後の__post_init__でチェック

    # * 画像保存時の設定
    snapshot_dpi: int
    extension: str

    # *出力画像の大きさ [cm]
    # （参考）A4用紙の縦向きサイズ（縦 × 横）は 29.7 × 21.0[cm]
    fig_horizontal_cm: float  # 横方向のみ設定．縦方向は自動で調節される
    tooning_for_fig: float

    # * 必須の物理量（x, y, disa）がどの素データのどの列か．0-index
    xydisa_file_path: str
    col_index_x: int
    col_index_y: int
    col_index_disa: int

    # * このファイルからの各相対パス
    path_list: list
    num_x_in_pathstr: int

    # * プロット関連
    timestep_ms: int
    snap_start_time_ms: int
    snap_end_time_ms: int
    xlim_min: float
    xlim_max: float
    ylim_min: float
    ylim_max: float

    # * アニメーション関連
    framerate: int
    crf_num: int

    # *全体の見た目の設定
    axis_lw: float  # 軸線の太さ
    is_plot_axis_bottom: bool  # 下側の軸を表示するか
    is_plot_axis_left: bool  # 左側の軸を表示するか
    is_plot_axis_top: bool  # 上側の軸を表示するか
    is_plot_axis_right: bool  # 右側の軸を表示するか
    is_plot_ticks_bottom: bool  # 下側のx軸の目盛りを表示
    is_plot_ticks_left: bool  # 左側のy軸の目盛りを表示
    is_plot_ticks_top: bool  # 上側のx軸の目盛りを非表示
    is_plot_ticks_right: bool  # 右側のy軸の目盛りを非表示

    # *フォント関連
    base_font_size: float  # 基準フォントサイズ
    xlabel_font_size: float  # x軸タイトルのフォントサイズ
    ylabel_font_size: float  # y軸 タイトルのフォントサイズ
    xticks_font_size: float  # x軸目盛りの値のフォントサイズ
    yticks_font_size: float  # y軸目盛りの値のフォントサイズ
    timetext_font_size: float  # 時刻テキストのフォントサイズ
    colorbar_title_font_size: float  # カラーバーのタイトルのサイズ
    colorbar_ticks_font_size: float  # カラーバーの目盛りの値のサイズ
    reference_vector_font_size: float  # reference vectorのラベルのフォントサイズ
    is_use_TimesNewRoman_in_mathtext: bool  # 数式で可能な限りTimes New Romanを使うか（FalseでTeXっぽいフォントを使う）

    # *目盛りの設定（x軸）
    # -主目盛り-
    anchor_x_ticks: float  # 主目盛りで必ず表示する座標
    space_x_ticks: float  # 主目盛りの間隔
    strformatter_x: (
        str | None
    )  # 主目盛りの値の書式等を変更したいときにいじる（変更しない場合はNoneにする）．
    x_tickslabel_pad: float  # x軸主目盛りから目盛りラベルをどれだけ離すか
    x_ticks_length: float  # x軸主目盛り線の長さ
    x_ticks_width: float  # x軸主目盛り線の線幅
    # -副目盛り-
    is_plot_mticks_x: bool  # 副目盛りをプロットするか
    num_x_mtick: int  # 副目盛りの数
    x_mticks_length: float  # x軸補助目盛り線の長さ
    x_mticks_width: float  # x軸補助目盛り線の線幅

    # *目盛りの設定（y軸）
    # -主目盛り-
    anchor_y_ticks: float  # 主目盛りで必ず表示する座標
    space_y_ticks: float  # 主目盛りの間隔
    strformatter_y: (
        str | None
    )  # 主目盛りの値の書式等を変更したいときにいじる（変更しない場合はNoneにする）．
    y_tickslabel_pad: float  # y軸主目盛りから目盛りラベルをどれだけ離すか
    y_ticks_length: float  # y軸主目盛り線の長さ
    y_ticks_width: float  # y軸主目盛り線の線幅
    # -副目盛り-
    is_plot_mticks_y: bool  # 副目盛りをプロットするか
    num_y_mtick: int  # 副目盛りの数
    y_mticks_length: float  # y軸補助目盛り線の長さ
    y_mticks_width: float  # y軸補助目盛り線の線幅

    # *軸ラベルの設定（x軸）
    xlabel_text: str  # ラベルのテキスト
    xlabel_pos: float  # テキストの中心のx座標（データの単位）
    xlabel_offset: float

    # *軸ラベルの設定（y軸）
    ylabel_text: str  # ラベルのテキスト
    ylabel_pos: float  # テキストの中心のy座標（データ単位）
    y_horizontalalignment: (
        str  # テキストのx方向の配置の基準（基本は"center" or "right"）
    )
    ylabel_offset: float
    is_horizontal_ylabel: bool  # ラベルを横向きにするか

    # * 時刻のテキストの設定
    time_text: str  # 時刻テキスト（xxxxxの部分に時刻が挿入される）
    strformatter_timetext: str  # 時刻テキストの書式（小数点以下のゼロ埋めの調節で使用）
    time_text_pos_x: float  # 時刻テキストの左端のx座標（データの単位）
    time_text_pos_y: float  # 時刻テキストの下端のy座標（データの単位）

    # * カラーバー関連の設定
    is_horizontal_colorbar: bool  # カラーバーを横向きにするか
    colorbar_pad: float  # TODO  カラーバーがプロットのボックスからどれだけ離れているか（チューニング）
    colorbar_labelpad: float  # カラーバーのタイトルをバーからどれだけ離すか
    colorbar_shrink: float  # カラーバーをどれくらい縮めるか（0より大きく1以下の値）
    colorbar_aspect: (
        float  # カラーバーのアスペクト比（shrinkを先に調整してからがよさそう）
    )
    axis_lw_colorbar: float  # カラーバーの枠線の太さ
    num_edges_colorbar: int  # カラーバーの区切りの数
    is_drawedges_colorbar: bool  # カラーバーの区切り線を描画するか
    # -主目盛り-
    num_colorbar_ticks: int  # 主目盛りの数
    strformatter_colorbar: (
        str | None
    )  # 主目盛りの値の書式等を変更したいときにいじる（変更しない場合はnullにする）．
    colorbar_tickslabel_pad: float  # x軸主目盛りから目盛りラベルをどれだけ離すか
    colorbar_ticks_length: float  # x軸主目盛り線の長さ
    colorbar_ticks_width: float  # x軸主目盛り線の線幅
    # -副目盛り-
    is_plot_mticks_colorbar: bool  # 副目盛りをプロットするか
    num_colorbar_mtick: int  # 副目盛りの数
    colorbar_mticks_length: float  # x軸補助目盛り線の長さ
    colorbar_mticks_width: float  # x軸補助目盛り線の線幅

    # * svg出力用．これをTrueにした場合，ここで指定した時刻のsvgのみを作成する．jpegの画像やアニメーションの作成は行わないので注意．
    svg_flag: bool
    svg_snap_time_ms: int | None

    # * contourプロットでのグループ分けの選択
    grouping_id: str

    # * ベクトルプロットのid振り用（コード内部で直接使用することはない）
    vector_list: list

    # * ベクトルプロットの選択
    vector_id: str

    def __post_init__(self) -> None:
        class_dict = dataclasses.asdict(self)

        # * 1. 型チェック
        # self.__annotations__は {引数の名前:指定する型}
        for class_arg_name, class_arg_expected_type in self.__annotations__.items():
            if not isinstance(class_dict[class_arg_name], class_arg_expected_type):
                raise ValueError(
                    f"{class_arg_name}の型が一致しません．\n{class_arg_name}の型は現在は{type(class_dict[class_arg_name])}ですが，{class_arg_expected_type}である必要があります．"
                )

        # path_listの処理
        if not all([isinstance(name, str) for name in self.path_list]):
            raise ValueError(
                f"path_listの要素の型が一致しません．\npath_listの中身の型は全て{str}である必要があります．"
            )

        # plot_order_list_contourの処理
        if not all([isinstance(name, str) for name in self.plot_order_list_contour]):
            raise ValueError(
                f"plot_order_list_contourの要素の型が一致しません．\nplot_order_list_contourの中身の型は全て{str}である必要があります．"
            )

        # plot_order_list_groupの処理
        if not all([isinstance(name, str) for name in self.plot_order_list_group]):
            raise ValueError(
                f"plot_order_list_groupの要素の型が一致しません．\nplot_order_list_groupの中身の型は全て{str}である必要があります．"
            )

        # vector_listの処理
        if not all([isinstance(name, str) for name in self.vector_list]):
            raise ValueError(
                f"vector_listの要素の型が一致しません．\nvector_listの中身の型は全て{str}である必要があります．"
            )

        print("1. 型チェック OK")

        # svg関連の処理
        if self.svg_flag and self.svg_snap_time_ms is None:
            raise ValueError(
                "svg_snap time_msの値が設定されていません．svgを作成したい時刻[ms]を設定してください"
            )

        # * 2.諸々のエラー処理
        # plot_order_list_contourの処理
        if "NOT_PLOT_BELOW" not in self.plot_order_list_contour:
            raise ValueError(
                "plot_order_list_contour内に'NOT_PLOT_BELOW'という文字列を含めてください"
            )
        # plot_order_list_groupの処理
        if "NOT_PLOT_BELOW" not in self.plot_order_list_group:
            raise ValueError(
                "plot_order_list_group内に'NOT_PLOT_BELOW'という文字列を含めてください"
            )

        print("IN_PARAMS construct OK")
        return


@dataclasses.dataclass(frozen=True)
class DataclassContour:
    label: str
    data_file_path: str
    col_index: int
    min_value_contour: float
    max_value_contour: float
    cmap: str
    is_plot_vector: bool

    def __post_init__(self) -> None:
        class_dict = dataclasses.asdict(self)
        # * 1. 型チェック
        # self.__annotations__は {引数の名前:指定する型}
        for class_arg_name, class_arg_expected_type in self.__annotations__.items():
            if not isinstance(class_dict[class_arg_name], class_arg_expected_type):
                raise ValueError(
                    f"{class_arg_name}の型が一致しません．\n{class_arg_name}の型は現在は{type(class_dict[class_arg_name])}ですが，{class_arg_expected_type}である必要があります．"
                )

        print("CUR_CONTOUR_PARAMS construct OK")


@dataclasses.dataclass(frozen=True)
class DataclassVector:
    col_index_vectorx: int
    col_index_vectory: int
    scaler_length_vector: float
    scaler_width_vector: float
    headlength_vector: float
    headaxislength_vector: float
    headwidth_vector: float
    length_reference_vector: float
    reference_vector_text: str
    strformatter_reference_vector_text: str
    reference_vector_pos_x: float
    reference_vector_pos_y: float
    reference_vector_labelpad: float

    def __post_init__(self) -> None:
        class_dict = dataclasses.asdict(self)
        # * 1. 型チェック
        # self.__annotations__は {引数の名前:指定する型}
        for class_arg_name, class_arg_expected_type in self.__annotations__.items():
            if not isinstance(class_dict[class_arg_name], class_arg_expected_type):
                raise ValueError(
                    f"{class_arg_name}の型が一致しません．\n{class_arg_name}の型は現在は{type(class_dict[class_arg_name])}ですが，{class_arg_expected_type}である必要があります．"
                )

        print("VECTOR_PARAMS construct OK")


@dataclasses.dataclass(frozen=True)
class DataclassGroupConfig:
    data_file_path: str  # データがあるファイルのpythonファイルからの相対パス
    col_index: int  # データが何列目にあるか　（0-index）

    def __post_init__(self) -> None:
        class_dict = dataclasses.asdict(self)
        # * 1. 型チェック
        # self.__annotations__は {引数の名前:指定する型}
        for class_arg_name, class_arg_expected_type in self.__annotations__.items():
            if not isinstance(class_dict[class_arg_name], class_arg_expected_type):
                raise ValueError(
                    f"{class_arg_name}の型が一致しません．\n{class_arg_name}の型は現在は{type(class_dict[class_arg_name])}ですが，{class_arg_expected_type}である必要があります．"
                )

        print("GROUP_CONFIG_PARAMS construct OK")


@dataclasses.dataclass(frozen=True)
class DataclassGroupEachIndex:
    label: str
    group_color: str
    group_alpha: float
    group_is_plot_vector: bool
    contour_color: str | None
    contour_alpha: float
    contour_is_plot_vector: bool
    particle_zorder: float
    vector_zorder: float

    def __post_init__(self) -> None:
        class_dict = dataclasses.asdict(self)
        # * 1. 型チェック
        # self.__annotations__は {引数の名前:指定する型}
        for class_arg_name, class_arg_expected_type in self.__annotations__.items():
            if not isinstance(class_dict[class_arg_name], class_arg_expected_type):
                raise ValueError(
                    f"{class_arg_name}の型が一致しません．\n{class_arg_name}の型は現在は{type(class_dict[class_arg_name])}ですが，{class_arg_expected_type}である必要があります．"
                )

        print(f"GROUPIDX_PARAMS construct OK({self.label})")


def read_inparam_yaml_as_dict() -> Dict:
    input_yaml_name = sys.argv[1]

    with open(
        Path(__file__).parent / "input_yaml" / f"{input_yaml_name}",
        mode="r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def construct_input_parameters_dataclass() -> DataclassInputParameters:
    inparam_dict = read_inparam_yaml_as_dict()

    del inparam_dict["contour"]
    del inparam_dict["group"]
    del inparam_dict["vector"]

    return DataclassInputParameters(**inparam_dict)


def construct_contour_dataclass(
    contour_name: str | None = None,
) -> DataclassContour | None:
    if contour_name is None:
        return None

    inparam_dict = read_inparam_yaml_as_dict()

    return DataclassContour(**inparam_dict["contour"][contour_name])


def construct_vector_dataclass() -> DataclassVector:
    inparam_dict = read_inparam_yaml_as_dict()

    return DataclassVector(**inparam_dict["vector"][IN_PARAMS.vector_id])


def construct_group_config_dataclass(group_name: str) -> DataclassGroupConfig:
    inparam_dict = read_inparam_yaml_as_dict()

    return DataclassGroupConfig(**inparam_dict["group"][group_name]["config"])


def construct_dict_groupindex_to_groupeachidxdataclass(
    group_name: str,
) -> Dict[int, DataclassGroupEachIndex]:
    inparam_dict = read_inparam_yaml_as_dict()
    groupidx_to_paramsdict: Dict[int, Dict[str, Any]] = inparam_dict["group"][
        group_name
    ]["each_idx"]

    res_dict = dict()
    for group_idx, data_dict in groupidx_to_paramsdict.items():
        res_dict[group_idx] = DataclassGroupEachIndex(**data_dict)

    return res_dict


def set_mplparams_init() -> None:
    # 描画高速化
    mplstyle.use("fast")

    # 以下はイラレで編集可能なsvgを出力するために必要
    mpl.use("Agg")
    plt.rcParams["svg.fonttype"] = "none"

    # 軸設定
    plt.rcParams["xtick.direction"] = "out"  # x軸の目盛りの向き
    plt.rcParams["ytick.direction"] = "out"  # y軸の目盛りの向き
    plt.rcParams["xtick.minor.visible"] = True  # x軸補助目盛りの追加
    plt.rcParams["ytick.minor.visible"] = True  # y軸補助目盛りの追加

    # 余白の自動調整
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05

    # MatplotlibのデフォルトフォントをTimes New Romanに設定
    plt.rcParams["font.family"] = "Times New Roman"

    # mathtext関連
    if IN_PARAMS.is_use_TimesNewRoman_in_mathtext:
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.it"] = "Times New Roman:italic"
        plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
        plt.rcParams["mathtext.bfit"] = "Times New Roman:italic:bold"
        plt.rcParams["mathtext.rm"] = "Times New Roman"
        plt.rcParams["mathtext.fallback"] = "cm"
    else:
        plt.rcParams["mathtext.fontset"] = "cm"

    # 全体の見た目の設定（軸と目盛りの表示）
    plt.rcParams["axes.spines.bottom"] = (
        IN_PARAMS.is_plot_axis_bottom
    )  # 下側の軸を表示するか
    plt.rcParams["axes.spines.left"] = (
        IN_PARAMS.is_plot_axis_left
    )  # 左側の軸を表示するか
    plt.rcParams["axes.spines.top"] = IN_PARAMS.is_plot_axis_top  # 上側の軸を表示するか
    plt.rcParams["axes.spines.right"] = (
        IN_PARAMS.is_plot_axis_right
    )  # 右側の軸を表示するか
    plt.rcParams["xtick.bottom"] = (
        IN_PARAMS.is_plot_ticks_bottom
    )  # 下側のx軸の目盛りを表示
    plt.rcParams["ytick.left"] = IN_PARAMS.is_plot_ticks_left  # 左側のy軸の目盛りを表示
    plt.rcParams["xtick.top"] = IN_PARAMS.is_plot_ticks_top  # 上側のx軸の目盛りを非表示
    plt.rcParams["ytick.right"] = (
        IN_PARAMS.is_plot_ticks_right
    )  # 右側のy軸の目盛りを非表示

    # 目盛り関係
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    # 軸関係
    plt.rcParams["axes.linewidth"] = IN_PARAMS.axis_lw
    plt.rcParams["xtick.minor.visible"] = IN_PARAMS.is_plot_mticks_x
    plt.rcParams["ytick.minor.visible"] = IN_PARAMS.is_plot_mticks_y

    # 凡例の見た目設定（今は使用なし）
    plt.rcParams["legend.fancybox"] = False  # 丸角OFF
    plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更


def get_mask_array_by_plot_region(snap_time_ms: int) -> NDArray[np.bool_]:
    original_data = np.loadtxt(
        Path(__file__).parent
        / Path(
            IN_PARAMS.xydisa_file_path.replace(
                "x" * IN_PARAMS.num_x_in_pathstr,
                f"{snap_time_ms:0{IN_PARAMS.num_x_in_pathstr}}",
            )
        ),
        usecols=(
            IN_PARAMS.col_index_x,
            IN_PARAMS.col_index_y,
            IN_PARAMS.col_index_disa,
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
    usecols: tuple[int, ...] | int,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    original_data = np.loadtxt(
        Path(__file__).parent
        / Path(
            IN_PARAMS.xydisa_file_path.replace(
                "x" * IN_PARAMS.num_x_in_pathstr,
                f"{snap_time_ms:0{IN_PARAMS.num_x_in_pathstr}}",
            )
        ),
        usecols=usecols,
        dtype=np.float64,
    )

    masked_data = original_data[mask_array]

    if mask_array_by_group is not None:
        masked_data = masked_data[mask_array_by_group]

    return masked_data.T


def get_group_index_array(
    snap_time_ms: int, mask_array: NDArray[np.bool_]
) -> NDArray[np.int8]:
    masked_group_index = np.loadtxt(
        Path(__file__).parent
        / Path(
            GROUP_CONFIG_PARAMS.data_file_path.replace(
                "x" * IN_PARAMS.num_x_in_pathstr,
                f"{snap_time_ms:0{IN_PARAMS.num_x_in_pathstr}}",
            )
        ),
        dtype=np.int8,
        usecols=GROUP_CONFIG_PARAMS.col_index,
    )[mask_array]

    return masked_group_index


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
    diameter_in_data_units: NDArray[np.float64], fig: Figure, axis: Axes
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
    fig: Figure,
    ax: Axes,
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
        clip_on=not IN_PARAMS.svg_flag,
    )

    return


def get_norm_for_color_contour() -> Normalize:
    assert CUR_CONTOUR_PARAMS is not None
    return Normalize(
        vmin=CUR_CONTOUR_PARAMS.min_value_contour,
        vmax=CUR_CONTOUR_PARAMS.max_value_contour,
    )


def get_cmap_for_color_contour() -> Colormap:
    assert CUR_CONTOUR_PARAMS is not None
    cmap_name = CUR_CONTOUR_PARAMS.cmap

    try:
        if cmap_name == "small rainbow":
            return LinearSegmentedColormap.from_list(
                "custom",
                ["blue", "cyan", "lime", "yellow", "red"],
                N=IN_PARAMS.num_edges_colorbar,
            )
        else:
            return plt.get_cmap(cmap_name, IN_PARAMS.num_edges_colorbar)

    except ValueError:
        raise ValueError(
            f'cmap で設定されているcolormap名（{cmap_name}）は存在しません．\nhttps://matplotlib.org/stable/users/explain/colors/colormaps.html に載っているcolormap名，もしくは"small rainbow"を設定してください．'
        )


def get_facecolor_by_physics_contour(
    plot_name: str,
    snap_time_ms: int,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    assert CUR_CONTOUR_PARAMS is not None

    cmap = get_cmap_for_color_contour()
    norm = get_norm_for_color_contour()

    par_physics = load_par_data_masked_by_plot_region(
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=CUR_CONTOUR_PARAMS.col_index,
        mask_array_by_group=mask_array_by_group,
    )

    par_color = cmap(norm(par_physics))

    return par_color


def plot_colorbar(fig: Figure, ax: Axes, plot_name: str) -> None:
    assert CUR_CONTOUR_PARAMS is not None

    cmap = get_cmap_for_color_contour()
    norm = get_norm_for_color_contour()
    mappable = ScalarMappable(cmap=cmap, norm=norm)

    assert norm.vmin is not None and norm.vmax is not None

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=IN_PARAMS.colorbar_shrink,
        aspect=IN_PARAMS.colorbar_aspect,
        orientation="horizontal" if IN_PARAMS.is_horizontal_colorbar else "vertical",
        pad=IN_PARAMS.colorbar_pad,
        location="top",
        anchor=(0.5, 0.0),
        panchor=(0, 1.0),
        ticks=ticker.LinearLocator(numticks=IN_PARAMS.num_colorbar_ticks),
        format=IN_PARAMS.strformatter_colorbar,
        drawedges=IN_PARAMS.is_drawedges_colorbar,
    )

    # ラベルと目盛りの位置
    cbar.ax.xaxis.set_ticks_position("bottom")

    # ラベルの大きさ
    cbar.set_label(
        f"{CUR_CONTOUR_PARAMS.label}",
        fontsize=IN_PARAMS.colorbar_title_font_size,
        labelpad=IN_PARAMS.colorbar_labelpad,
    )

    # カラーバーの枠線の太さ
    cbar.outline.set_linewidth(IN_PARAMS.axis_lw_colorbar)

    # 主目盛り
    cbar.ax.tick_params(
        which="major",
        labelsize=IN_PARAMS.colorbar_ticks_font_size,
        pad=IN_PARAMS.colorbar_tickslabel_pad,
        length=IN_PARAMS.colorbar_ticks_length,
        width=IN_PARAMS.colorbar_ticks_width,
    )
    # 副目盛り
    if IN_PARAMS.is_plot_mticks_colorbar:
        cbar.minorticks_on()
        cbar.ax.tick_params(
            which="minor",
            length=IN_PARAMS.colorbar_mticks_length,
            width=IN_PARAMS.colorbar_mticks_width,
        )
        ax.xaxis.set_minor_locator(
            ticker.AutoMinorLocator(n=IN_PARAMS.num_colorbar_mtick + 1)
        )

    else:
        cbar.minorticks_off()

    return


def plot_velocity_vector(
    fig: Figure,
    ax: Axes,
    snap_time_ms: int,
    is_plot_reference_vector: bool,
    mask_array: NDArray[np.bool_],
    group_id_prefix: str,
    group_index: int,
    par_x: NDArray[np.float64],
    par_y: NDArray[np.float64],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> None:
    global list_extra_artists
    par_u, par_v = load_par_data_masked_by_plot_region(
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=(
            VECTOR_PARAMS.col_index_vectorx,
            VECTOR_PARAMS.col_index_vectory,
        ),
        mask_array_by_group=mask_array_by_group,
    )

    # scale_units="x"で軸の長さが基準
    # scale=10で，軸単位で0.1の長さが大きさ1のベクトルの長さに対応する
    original_scale = 10 / (IN_PARAMS.xlim_max - IN_PARAMS.xlim_min)
    scale = original_scale / VECTOR_PARAMS.scaler_length_vector
    width = original_scale / 5000 * VECTOR_PARAMS.scaler_width_vector
    q = ax.quiver(
        par_x,
        par_y,
        par_u,
        par_v,
        scale=scale,
        scale_units="x",
        width=width,
        headlength=VECTOR_PARAMS.headlength_vector,
        headaxislength=VECTOR_PARAMS.headaxislength_vector,
        headwidth=VECTOR_PARAMS.headwidth_vector,
        gid=f"{group_id_prefix}{group_index}",
        clip_on=not IN_PARAMS.svg_flag,
    )

    if is_plot_reference_vector:
        qk = ax.quiverkey(
            Q=q,
            X=VECTOR_PARAMS.reference_vector_pos_x,
            Y=VECTOR_PARAMS.reference_vector_pos_y,
            U=VECTOR_PARAMS.length_reference_vector,
            label=VECTOR_PARAMS.reference_vector_text.replace(
                "xxxxx",
                f"{VECTOR_PARAMS.length_reference_vector:{VECTOR_PARAMS.strformatter_reference_vector_text}}",
            ),
            fontproperties={"size": IN_PARAMS.reference_vector_font_size},
            labelsep=VECTOR_PARAMS.reference_vector_labelpad,
            labelpos="N",
            coordinates="data",
        )

        # 以下でreference vectorが余白の自動調節に含まれるようにする
        # TODO（labelpos="N"の場合のみ）

        # 1. Bboxを取得（表示座標系）
        fig.canvas.draw()  # 描画が完了している必要がある
        renderer = fig.canvas.get_renderer()
        bbox_text = qk.text.get_window_extent(renderer=renderer)

        # 2. データ座標系への変換
        inv = ax.transData.inverted()
        x0_data, y0_data = inv.transform((bbox_text.x0, bbox_text.y0))  # 左下隅
        x1_data, y1_data = inv.transform((bbox_text.x1, bbox_text.y1))  # 右上隅

        # 3. 隅にテキストを配置してこいつで自動調節をする
        ax.text(
            s="Dummy1",
            x=x0_data,
            y=y0_data,
            horizontalalignment="center",
            verticalalignment="center",
            alpha=0,
            gid="This is dummy text1.",
        )
        ax.text(
            s="Dummy2",
            x=x1_data,
            y=y1_data,
            horizontalalignment="center",
            verticalalignment="center",
            alpha=0,
            gid="This is dummy text2.",
        )
        ax.text(
            s="Dummy3",
            x=x0_data,
            y=y1_data,
            horizontalalignment="center",
            verticalalignment="center",
            alpha=0,
            gid="This is dummy text3.",
        )
        ax.text(
            s="Dummy4",
            x=x1_data,
            y=y0_data,
            horizontalalignment="center",
            verticalalignment="center",
            alpha=0,
            gid="This is dummy text4.",
        )

    return


def set_ax_lim(ax: Axes) -> None:
    ax.set_xlim(IN_PARAMS.xlim_min, IN_PARAMS.xlim_max)
    ax.set_ylim(IN_PARAMS.ylim_min, IN_PARAMS.ylim_max)

    return


def set_ax_xticks(ax: Axes) -> None:
    ax.xaxis.set_major_locator(
        ticker.MultipleLocator(
            base=IN_PARAMS.space_x_ticks, offset=IN_PARAMS.anchor_x_ticks
        )
    )

    if IN_PARAMS.strformatter_x is not None:
        ax.xaxis.set_major_formatter(
            ticker.FormatStrFormatter(IN_PARAMS.strformatter_x)
        )

    if IN_PARAMS.is_plot_mticks_x:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=IN_PARAMS.num_x_mtick + 1))

    # 主目盛り
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=IN_PARAMS.xticks_font_size,
        pad=IN_PARAMS.x_tickslabel_pad,
        length=IN_PARAMS.x_ticks_length,
        width=IN_PARAMS.x_ticks_width,
    )

    # 副目盛り
    ax.tick_params(
        axis="x",
        which="minor",
        length=IN_PARAMS.x_mticks_length,
        width=IN_PARAMS.x_mticks_width,
    )

    return


def set_ax_yticks(ax: Axes) -> None:
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(
            base=IN_PARAMS.space_y_ticks, offset=IN_PARAMS.anchor_y_ticks
        )
    )

    if IN_PARAMS.strformatter_y is not None:
        ax.yaxis.set_major_formatter(
            ticker.FormatStrFormatter(IN_PARAMS.strformatter_y)
        )

    if IN_PARAMS.is_plot_mticks_y:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=IN_PARAMS.num_y_mtick + 1))

    # 主目盛り
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=IN_PARAMS.yticks_font_size,
        pad=IN_PARAMS.y_tickslabel_pad,
        length=IN_PARAMS.y_ticks_length,
        width=IN_PARAMS.y_ticks_width,
    )

    # 副目盛り
    ax.tick_params(
        axis="y",
        which="minor",
        length=IN_PARAMS.y_mticks_length,
        width=IN_PARAMS.y_mticks_width,
    )

    return


def set_xlabel(ax: Axes) -> None:
    ax.text(
        s=IN_PARAMS.xlabel_text,
        x=IN_PARAMS.xlabel_pos,
        y=IN_PARAMS.ylim_min + IN_PARAMS.xlabel_offset,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=IN_PARAMS.xlabel_font_size,
        gid="x_title_text",
    )
    return


def set_ylabel(ax: Axes) -> None:
    tmp = ax.text(
        s=IN_PARAMS.ylabel_text,
        y=IN_PARAMS.ylabel_pos,
        x=IN_PARAMS.xlim_min + IN_PARAMS.ylabel_offset,
        verticalalignment="center",
        horizontalalignment=IN_PARAMS.y_horizontalalignment,
        fontsize=IN_PARAMS.ylabel_font_size,
        gid="y_title_text",
    )

    if not IN_PARAMS.is_horizontal_ylabel:
        tmp.set_rotation(90.0)

    return


def set_ax_time_text(ax: Axes, snap_time_ms: int) -> None:
    ax.text(
        s=IN_PARAMS.time_text.replace(
            "xxxxx", f"{snap_time_ms/1000:{IN_PARAMS.strformatter_timetext}}"
        ),
        x=IN_PARAMS.time_text_pos_x,
        y=IN_PARAMS.time_text_pos_y,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=IN_PARAMS.timetext_font_size,
        gid="time_text",
    )

    return


def change_facecolor_and_alpha_by_groupidxparams(
    par_color_masked_by_group: NDArray[np.float64],
    group_index: int,
) -> NDArray[np.float64]:
    assert CUR_CONTOUR_PARAMS is not None

    change_contour_color = GROUPIDX_PARAMS[group_index].contour_color
    change_contour_alpha = GROUPIDX_PARAMS[group_index].contour_alpha

    if change_contour_color is None:
        par_color_masked_by_group[:, 3] = change_contour_alpha  # ４列目がrgbaのa
    else:
        par_color_masked_by_group = np.full_like(
            par_color_masked_by_group,
            to_rgba(c=change_contour_color, alpha=change_contour_alpha),
        )

    return par_color_masked_by_group


def get_facecolor_array_for_contour(
    group_index: int,
    plot_name: str,
    snap_time_ms: int,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_],
) -> NDArray[np.float64]:
    par_color = get_facecolor_by_physics_contour(
        plot_name=plot_name,
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        mask_array_by_group=mask_array_by_group,
    )

    return change_facecolor_and_alpha_by_groupidxparams(
        par_color_masked_by_group=par_color,
        group_index=group_index,
    )


def get_facecolor_array_for_group(
    group_index: int,
    num_par_cur_group: int,
) -> NDArray[np.float64]:
    return np.full(
        (num_par_cur_group, 4),
        to_rgba(
            c=GROUPIDX_PARAMS[group_index].group_color,
            alpha=GROUPIDX_PARAMS[group_index].group_alpha,
        ),
    )


def get_cur_is_plot_vector(is_group_plot: bool, group_index: int) -> bool:
    if is_group_plot:
        return GROUPIDX_PARAMS[group_index].group_is_plot_vector

    else:
        assert CUR_CONTOUR_PARAMS is not None

        return (
            GROUPIDX_PARAMS[group_index].contour_is_plot_vector
            and CUR_CONTOUR_PARAMS.is_plot_vector
        )


def make_snap_each_snap_time(
    fig: Figure,
    ax: Axes,
    snap_time_ms: int,
    save_dir_snap_path: Path,
    plot_name: str,
    is_group_plot: bool,
) -> None:
    if not is_group_plot:
        assert CUR_CONTOUR_PARAMS is not None

    set_ax_lim(ax=ax)
    set_ax_xticks(ax=ax)
    set_ax_yticks(ax=ax)
    set_xlabel(ax=ax)
    set_ylabel(ax=ax)
    set_ax_time_text(ax=ax, snap_time_ms=snap_time_ms)

    # TODO キャンバス更新
    if snap_time_ms == IN_PARAMS.snap_start_time_ms:
        fig.canvas.draw()

    mask_array = get_mask_array_by_plot_region(snap_time_ms=snap_time_ms)

    par_group_index = get_group_index_array(
        snap_time_ms=snap_time_ms, mask_array=mask_array
    )

    particle_group_id_prefix = "particle"
    vector_group_id_prefix = "vector"

    is_plot_reference_vector = True
    for group_index in np.unique(par_group_index)[::-1]:
        mask_array_by_group: NDArray[np.bool_] = par_group_index == group_index

        # x, y, disaの取得
        par_x, par_y, par_disa = load_par_data_masked_by_plot_region(
            snap_time_ms=snap_time_ms,
            mask_array=mask_array,
            usecols=(
                IN_PARAMS.col_index_x,
                IN_PARAMS.col_index_y,
                IN_PARAMS.col_index_disa,
            ),
            mask_array_by_group=mask_array_by_group,
        )

        # 粒子の色の取得
        if is_group_plot:
            par_color = get_facecolor_array_for_group(
                group_index=group_index, num_par_cur_group=par_x.shape[0]
            )
        else:
            par_color = get_facecolor_array_for_contour(
                group_index=group_index,
                plot_name=plot_name,
                snap_time_ms=snap_time_ms,
                mask_array=mask_array,
                mask_array_by_group=mask_array_by_group,
            )

        # 粒子のプロット
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

        # ベクトルのプロット
        if get_cur_is_plot_vector(is_group_plot=is_group_plot, group_index=group_index):
            plot_velocity_vector(
                fig=fig,
                ax=ax,
                snap_time_ms=snap_time_ms,
                mask_array=mask_array,
                is_plot_reference_vector=is_plot_reference_vector,
                mask_array_by_group=mask_array_by_group,
                par_x=par_x,
                par_y=par_y,
                group_id_prefix=vector_group_id_prefix,
                group_index=group_index,
            )
            # 初回のみreference vectorをプロット
            is_plot_reference_vector = False

    # 以下，画像の保存処理

    save_file_name_without_extension = (
        f"snap{snap_time_ms:0{IN_PARAMS.num_x_in_pathstr}}_{plot_name}"
    )
    save_file_path = save_dir_snap_path / Path(
        f"{save_file_name_without_extension}.{IN_PARAMS.extension}"
    )
    fig.savefig(save_file_path)

    plt.cla()

    return


def make_snap_all_snap_time(
    snap_time_array_ms: NDArray[np.int64],
    save_dir_sub_path: Path,
    plot_name: str,
    is_group_plot: bool,
) -> None:
    if not is_group_plot:
        # CUR_CONTOUR_PARAMSを現在のcontourプロット用の情報に更新
        global CUR_CONTOUR_PARAMS
        CUR_CONTOUR_PARAMS = construct_contour_dataclass(contour_name=plot_name)
        assert CUR_CONTOUR_PARAMS is not None

    save_dir_snap_path = save_dir_sub_path / "snap_shot"
    save_dir_snap_path.mkdir(exist_ok=True, parents=True)

    scaler_cm_to_inch = 1 / 2.54
    disired_fig_width = IN_PARAMS.fig_horizontal_cm * scaler_cm_to_inch

    fig = plt.figure(
        figsize=(
            disired_fig_width,
            5 * disired_fig_width,  # 縦方向が見切れないよう十分大きく取る
        ),
        dpi=IN_PARAMS.snapshot_dpi,
    )
    ax = fig.add_axes(
        (
            (0, 0, 1 * IN_PARAMS.tooning_for_fig, 1)
        ),  # widthはちょうど指定した大きさに近づくようチューニングする
        aspect="equal",
    )

    if not is_group_plot:
        plot_colorbar(fig=fig, ax=ax, plot_name=plot_name)

    # 本番プロット（調整のため最初だけ一回多くプロットしている）
    for snap_time_ms in np.insert(
        snap_time_array_ms, 0, [IN_PARAMS.snap_start_time_ms]
    ):
        try:
            make_snap_each_snap_time(
                fig=fig,
                ax=ax,
                snap_time_ms=snap_time_ms,
                save_dir_snap_path=save_dir_snap_path,
                plot_name=plot_name,
                is_group_plot=is_group_plot,
            )
            print(f"{snap_time_ms/1000:.03f} s {plot_name} plot finished")
        except FileNotFoundError:
            print(
                f"{snap_time_ms/1000:.03f} s時点の計算データがありません．スナップショットの作成を終了します．\n"
            )
            break

    print(f"{snap_time_ms/1000:.03f} s {plot_name} all make snap finished\n")
    plt.close()
    return


def make_animation_from_snap(
    snap_time_array_ms: NDArray[np.int64], save_dir_sub_path: Path, plot_name: str
) -> None:
    save_dir_animation_path = save_dir_sub_path / "animation"
    save_dir_animation_path.mkdir(exist_ok=True)

    # 連番でない画像の読み込みに対応させるための準備
    for_ffmpeg = []
    for snap_time_ms in snap_time_array_ms:
        cur_snap_path = (
            save_dir_sub_path / "snap_shot" / f"snap{snap_time_ms:05}_{plot_name}.jpeg"
        )
        # アニメーション作成で使うsnapが存在するかの確認
        if not cur_snap_path.exists():
            break
        for_ffmpeg.append(f"file 'snap{snap_time_ms:05}_{plot_name}.jpeg'")

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

    cur_save_file_name = f"{plot_name}.mp4"
    subprocess.run(
        cmd_list1 + cmd_list2 + [str(save_dir_animation_path / cur_save_file_name)],
        cwd=str(save_dir_sub_path / "snap_shot"),
    )

    # 以下は低画質用
    cur_save_file_name = f"{plot_name}_lowquality.mp4"
    subprocess.run(
        cmd_list1
        + ["-crf", f"{IN_PARAMS.crf_num}"]  # ここで動画の品質を調整
        + cmd_list2
        + [str(save_dir_animation_path / cur_save_file_name)],
        cwd=str(save_dir_sub_path / "snap_shot"),
    )

    # tmp_for_ffmpeg.txtを削除
    save_file_forffmpeg_path.unlink()

    print(f"{plot_name} animation finished\n")

    return


def execute_plot_contour_all() -> None:
    # スナップやアニメーションを保存するディレクトリ名
    save_dir_path: Path = Path(__file__).parent / IN_PARAMS.save_dir_name

    # スナップショットを出力する時間[ms]のarray
    snap_time_array_ms: NDArray[np.int64] = np.arange(
        IN_PARAMS.snap_start_time_ms,
        IN_PARAMS.snap_end_time_ms + IN_PARAMS.timestep_ms,
        IN_PARAMS.timestep_ms,
    )

    for cur_contour_name in IN_PARAMS.plot_order_list_contour:
        if cur_contour_name == "NOT_PLOT_BELOW":
            break

        save_dir_sub_path = save_dir_path / cur_contour_name

        make_snap_all_snap_time(
            snap_time_array_ms=snap_time_array_ms,
            save_dir_sub_path=save_dir_sub_path,
            plot_name=cur_contour_name,
            is_group_plot=False,
        )
        make_animation_from_snap(
            snap_time_array_ms=snap_time_array_ms,
            save_dir_sub_path=save_dir_sub_path,
            plot_name=cur_contour_name,
        )
        print(f"contour:{cur_contour_name} finished\n")

    return


def execute_plot_group_all() -> None:
    # スナップやアニメーションを保存するディレクトリ名
    save_dir_path: Path = Path(__file__).parent / IN_PARAMS.save_dir_name

    # スナップショットを出力する時間[ms]のarray
    snap_time_array_ms: NDArray[np.int64] = np.arange(
        IN_PARAMS.snap_start_time_ms,
        IN_PARAMS.snap_end_time_ms + IN_PARAMS.timestep_ms,
        IN_PARAMS.timestep_ms,
    )

    for cur_group_name in IN_PARAMS.plot_order_list_group:
        if cur_group_name == "NOT_PLOT_BELOW":
            break

        save_dir_sub_path = save_dir_path / cur_group_name

        make_snap_all_snap_time(
            snap_time_array_ms=snap_time_array_ms,
            save_dir_sub_path=save_dir_sub_path,
            plot_name=cur_group_name,
            is_group_plot=True,
        )
        make_animation_from_snap(
            snap_time_array_ms=snap_time_array_ms,
            save_dir_sub_path=save_dir_sub_path,
            plot_name=cur_group_name,
        )
        print(f"group:{cur_group_name} finished\n")

    return


# ---グローバル変数群---
# contour, vector, group以外のすべてのyamlの設定を格納
IN_PARAMS = construct_input_parameters_dataclass()
# contourの設定を保持（contourプロットの最初で更新する）
CUR_CONTOUR_PARAMS = construct_contour_dataclass()
# vectorの設定を格納（vector_idで指定したもの）
VECTOR_PARAMS = construct_vector_dataclass()
# groupのうち，configの設定を格納
GROUP_CONFIG_PARAMS = construct_group_config_dataclass(group_name=IN_PARAMS.grouping_id)
# groupのうち，each_idx全ての「idx -> そのidx内の設定」の辞書
GROUPIDX_PARAMS = construct_dict_groupindex_to_groupeachidxdataclass(
    group_name=IN_PARAMS.grouping_id
)
# ---グローバル変数群---


def main() -> None:
    print(f"plot execute by {sys.argv[1]}\n")

    set_mplparams_init()

    # contourプロット
    execute_plot_contour_all()

    # groupプロット
    execute_plot_group_all()

    print("all finished")

    return


if __name__ == "__main__":
    print(mpl.matplotlib_fname())
    main()
