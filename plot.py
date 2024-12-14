import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
import numpy as np
import yaml  # type: ignore
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.quiver import QuiverKey
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
    is_make_snap: bool
    snap_dpi: int
    snap_extension: str

    # * アニメーション関連
    is_make_anim: bool
    anim_extension: str  # アニメーション作成に使用するスナップショットの拡張子（基本は，png or jpeg）
    framerate: int  # アニメーションのフレームレート．実スケールのものを作りたい時は 1000 / anim_timestep_ms を指定する
    crf_num: int  # 値が大きいほどlowqualityの動画の質が下がる．パワポに載せる動画など，ファイルサイズ小さめのが要るとき用にチューニング（2MB以下になるのがよいか）

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

    # * プロット関連
    snap_timestep_ms: int
    snap_start_time_ms: int
    snap_end_time_ms: int
    anim_timestep_ms: int
    anim_start_time_ms: int
    anim_end_time_ms: int

    xlim_min: float
    xlim_max: float
    ylim_min: float
    ylim_max: float

    # *全体の見た目の設定
    is_plot_hypervisor1: bool  # xy軸ラベル，時刻テキストなど，軸の表示範囲を変えると再度調整が必要なものの表示有無（最初にレイアウトを調整する時にはFalse，最後にTrueにして軸ラベルなどの位置調整を行うことを推奨）
    is_plot_hypervisor2: bool  # 軸線，目盛，目盛ラベルなどの表示有無（粒子やベクトルのみをプロットしたい時に使用）

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
    is_plot_xtickslabel_bottom: bool  # 下側のy軸の目盛ラベルを表示するか
    is_plot_xtickslabel_top: bool  # 上側のy軸の目盛ラベルを表示するか

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
    is_plot_ytickslabel_left: bool  # 左側のy軸の目盛ラベルを表示するか
    is_plot_ytickslabel_right: bool  # 右側のy軸の目盛ラベルを表示するか

    # -副目盛り-
    is_plot_mticks_y: bool  # 副目盛りをプロットするか
    num_y_mtick: int  # 副目盛りの数
    y_mticks_length: float  # y軸補助目盛り線の長さ
    y_mticks_width: float  # y軸補助目盛り線の線幅

    # *軸ラベルの設定（x軸）
    is_plot_xlabel_text: bool
    xlabel_text: str  # ラベルのテキスト
    xlabel_pos: float  # テキストの中心のx座標（データの単位）
    xlabel_offset: float

    # *軸ラベルの設定（y軸）
    is_plot_ylabel_text: bool
    ylabel_text: str  # ラベルのテキスト
    ylabel_pos: float  # テキストの中心のy座標（データ単位）
    ylabel_horizontalalignment: (
        str  # テキストのx方向の配置の基準（基本は"center" or "right"）
    )
    ylabel_offset: float
    is_horizontal_ylabel: bool  # ラベルを横向きにするか

    # * 時刻のテキストの設定
    is_plot_time_text: bool  # 時刻テキストをプロットするか
    time_text: str  # 時刻テキスト（xxxxxの部分に時刻が挿入される）
    strformatter_timetext: str  # 時刻テキストの書式（小数点以下のゼロ埋めの調節で使用）
    time_text_pos_x: float  # 時刻テキストの左端のx座標（データの単位）
    time_text_pos_y: float  # 時刻テキストの下端のy座標（データの単位）

    # * カラーバー関連の設定
    is_plot_colorbar: bool
    is_horizontal_colorbar: bool  # カラーバーを横向きにするか
    colorbar_pos: float  # カラーバーの位置の微調整（横向きならx方向，縦向きならy方向について，0.5でちょうどプロットのボックスの中心．0.0〜1.0 でプロットのボックスの領域内）
    colorbar_pad: float
    colorbar_shrink: float  # カラーバーをどれくらい縮めるか（0より大きく1以下の値）
    colorbar_aspect: (
        float  # カラーバーのアスペクト比（shrinkを先に調整してからがよさそう）
    )
    colorbar_label_horizontalalignment: (
        str  # テキストのx方向の配置の基準（"center" or "left" or "right"）
    )
    colorbar_label_verticalalignment: (
        str  # テキストのy方向の配置の基準（基本は"center" or "bottom" or "top"）
    )
    colorbar_label_x: float  # カラーバーのタイトルラベルのx方向の調整（0.0〜1.0 でカラーバーの領域内）
    colorbar_label_y: float  # カラーバーのタイトルラベルのy方向の調整（0.0〜1.0 でカラーバーの領域内）
    colorbar_rotation: float
    axis_lw_colorbar: float  # カラーバーの枠線の太さ
    num_edges_colorbar: int  # カラーバーの区切りの数
    is_drawedges_colorbar: bool  # カラーバーの区切り線を描画するか
    # -主目盛り-
    num_colorbar_ticks: int  # 主目盛りの数
    colorbar_tickslabel_pad: float  # x軸主目盛りから目盛りラベルをどれだけ離すか
    colorbar_ticks_length: float  # x軸主目盛り線の長さ
    colorbar_ticks_width: float  # x軸主目盛り線の線幅
    is_reverse_colorbar_tickslabel: (
        bool  # 目盛ラベルの位置をカラーバーに対して反転させるか
    )
    # -副目盛り-
    is_plot_mticks_colorbar: bool  # 副目盛りをプロットするか
    num_colorbar_mtick: int  # 副目盛りの数
    colorbar_mticks_length: float  # x軸補助目盛り線の長さ
    colorbar_mticks_width: float  # x軸補助目盛り線の線幅

    # * contourプロットでのグループ分けの選択
    grouping_id: str

    # * ベクトルプロットのid振り用（コード内部で直接使用することはない）
    vector_list: list

    # * ベクトルプロットの選択
    vector_id: str

    # * advanced setting
    scaler_s_to_ms: int
    num_x_in_pathstr: int

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
    strformatter_colorbar: str | None
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
    data_file_path: str
    col_index_vectorx: int
    col_index_vectory: int
    scaler_length_vector: float
    scaler_width_vector: float
    headlength_vector: float
    headaxislength_vector: float
    headwidth_vector: float

    is_plot_reference_vector: bool
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

        print("PLOT_VECTOR_PARAMS construct OK")


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


def construct_dict_of_contour_dataclass() -> Dict[str, DataclassContour]:
    inparam_dict_contour = read_inparam_yaml_as_dict()["contour"]

    res_dict = dict()
    for plot_contour_name in IN_PARAMS.plot_order_list_contour:
        if plot_contour_name == "NOT_PLOT_BELOW":
            break

        res_dict[plot_contour_name] = DataclassContour(
            **inparam_dict_contour[plot_contour_name]
        )

    return res_dict


def construct_vector_dataclass() -> DataclassVector:
    inparam_dict = read_inparam_yaml_as_dict()

    return DataclassVector(**inparam_dict["vector"][IN_PARAMS.vector_id])


def construct_dict_of_group_config_dataclass() -> Dict[str, DataclassGroupConfig]:
    inparam_dict_group = read_inparam_yaml_as_dict()["group"]

    res_dict: Dict[str, DataclassGroupConfig] = dict()

    for plot_group_name in IN_PARAMS.plot_order_list_group:
        if plot_group_name == "NOT_PLOT_BELOW":
            break

        res_dict[plot_group_name] = DataclassGroupConfig(
            **inparam_dict_group[plot_group_name]["config"]
        )

    return res_dict


def construct_dict_of_group_idx_dataclass() -> (
    Dict[str, Dict[int, DataclassGroupEachIndex]]
):
    inparam_dict_group = read_inparam_yaml_as_dict()["group"]

    res_dict: Dict[str, Dict[int, DataclassGroupEachIndex]] = dict()

    for plot_group_name in IN_PARAMS.plot_order_list_group:
        if plot_group_name == "NOT_PLOT_BELOW":
            break

        tmp_dict: Dict[int, DataclassGroupEachIndex] = {}
        cur_group_dict: Dict[int, Dict[str, Any]] = inparam_dict_group[plot_group_name][
            "each_idx"
        ]
        for group_idx, data_dict in cur_group_dict.items():
            tmp_dict[group_idx] = DataclassGroupEachIndex(**data_dict)

        res_dict[plot_group_name] = tmp_dict

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
        encoding="utf-8",
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
    pardata_filepath_str_time_replaced_by_xxxxx: str,
    snap_time_ms: int,
    usecols: tuple[int, ...] | int,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    original_data = np.loadtxt(
        Path(__file__).parent
        / Path(
            pardata_filepath_str_time_replaced_by_xxxxx.replace(
                "x" * IN_PARAMS.num_x_in_pathstr,
                f"{snap_time_ms:0{IN_PARAMS.num_x_in_pathstr}}",
            )
        ),
        usecols=usecols,
        dtype=np.float64,
        encoding="utf-8",
    )

    masked_data = original_data[mask_array]

    if mask_array_by_group is not None:
        masked_data = masked_data[mask_array_by_group]

    return masked_data.T


def get_group_index_array(
    snap_time_ms: int, mask_array: NDArray[np.bool_]
) -> NDArray[np.int32]:
    masked_group_index = np.loadtxt(
        Path(__file__).parent
        / Path(
            PLOT_GROUP_CONFIG_PARAMS[IN_PARAMS.grouping_id].data_file_path.replace(
                "x" * IN_PARAMS.num_x_in_pathstr,
                f"{snap_time_ms:0{IN_PARAMS.num_x_in_pathstr}}",
            )
        ),
        dtype=np.int32,
        usecols=PLOT_GROUP_CONFIG_PARAMS[IN_PARAMS.grouping_id].col_index,
        encoding="utf-8",
    )[mask_array]

    return masked_group_index


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

    groupingid = IN_PARAMS.grouping_id
    ax.scatter(
        par_x,
        par_y,
        s=s,
        c=par_color,
        linewidths=0,
        gid=f"{group_id_prefix}_{PLOT_GROUP_IDX_PARAMS[groupingid][group_index].label}",
        zorder=PLOT_GROUP_IDX_PARAMS[groupingid][group_index].particle_zorder,
    )

    return


def get_norm_for_color_contour(plot_name: str) -> Normalize:
    return Normalize(
        vmin=PLOT_CONTOUR_PARAMS[plot_name].min_value_contour,
        vmax=PLOT_CONTOUR_PARAMS[plot_name].max_value_contour,
    )


def get_cmap_for_color_contour(plot_name: str) -> Colormap:
    cmap_name = PLOT_CONTOUR_PARAMS[plot_name].cmap

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
    snap_time_ms: int,
    plot_name: str,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    cmap = get_cmap_for_color_contour(plot_name=plot_name)
    norm = get_norm_for_color_contour(plot_name=plot_name)

    par_physics = load_par_data_masked_by_plot_region(
        pardata_filepath_str_time_replaced_by_xxxxx=PLOT_CONTOUR_PARAMS[
            plot_name
        ].data_file_path,
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=PLOT_CONTOUR_PARAMS[plot_name].col_index,
        mask_array_by_group=mask_array_by_group,
    )

    par_color = cmap(norm(par_physics))

    return par_color


def plot_colorbar(fig: Figure, ax: Axes, plot_name: str) -> None:
    cmap = get_cmap_for_color_contour(plot_name=plot_name)
    norm = get_norm_for_color_contour(plot_name=plot_name)
    mappable = ScalarMappable(cmap=cmap, norm=norm)

    assert norm.vmin is not None and norm.vmax is not None

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=IN_PARAMS.colorbar_shrink,
        aspect=IN_PARAMS.colorbar_aspect,
        pad=IN_PARAMS.colorbar_pad,
        ticks=ticker.LinearLocator(numticks=IN_PARAMS.num_colorbar_ticks),
        format=PLOT_CONTOUR_PARAMS[plot_name].strformatter_colorbar,
        drawedges=IN_PARAMS.is_drawedges_colorbar,
        orientation="horizontal" if IN_PARAMS.is_horizontal_colorbar else "vertical",
        location="top" if IN_PARAMS.is_horizontal_colorbar else "right",
        anchor=(IN_PARAMS.colorbar_pos, 0.0)
        if IN_PARAMS.is_horizontal_colorbar
        else (0.0, IN_PARAMS.colorbar_pos),
        panchor=(0.0, 1.0) if IN_PARAMS.is_horizontal_colorbar else (1.0, 0.5),
    )

    # ラベルと目盛りの位置
    if IN_PARAMS.is_reverse_colorbar_tickslabel:
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.yaxis.set_ticks_position("right")
    else:
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.yaxis.set_ticks_position("left")

    # カラーバーのラベル
    cbar.ax.text(
        x=IN_PARAMS.colorbar_label_x,
        y=IN_PARAMS.colorbar_label_y,
        s=f"{PLOT_CONTOUR_PARAMS[plot_name].label}",
        transform=cbar.ax.transAxes,
        fontsize=IN_PARAMS.colorbar_title_font_size,
        horizontalalignment=IN_PARAMS.colorbar_label_horizontalalignment,
        verticalalignment=IN_PARAMS.colorbar_label_verticalalignment,
        rotation=IN_PARAMS.colorbar_rotation,
        gid="colorbar_label_text",
    )

    # カラーバーの枠線の太さ
    cbar.outline.set_linewidth(IN_PARAMS.axis_lw_colorbar)  # type: ignore

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


def update_refvec_corners(
    refvec_corners_for_dummytext: List[List[float]],
    fig: Figure,
    ax: Axes,
    qk: QuiverKey,
) -> None:
    # 1. Bboxを取得（表示座標系）
    fig.canvas.draw()  # 描画が完了している必要がある
    renderer = fig.canvas.get_renderer()  # type: ignore
    bbox_text = qk.text.get_window_extent(renderer=renderer)

    # 2. データ座標系への変換
    inv = ax.transData.inverted()
    x0_data, y0_data = inv.transform((bbox_text.x0, bbox_text.y0))  # 左下隅
    x1_data, y1_data = inv.transform((bbox_text.x1, bbox_text.y1))  # 右上隅

    # 参照渡しの辺りに注意
    refvec_corners_for_dummytext[0] = [x0_data, x1_data]
    refvec_corners_for_dummytext[1] = [y0_data, y1_data]

    return


def plot_transparent_dummytext_for_reference_vector_display(
    ax: Axes,
    refvec_corners_for_dummytext: List[List[float]],
) -> None:
    # 以下でreference vectorが余白の自動調節に含まれるようにする
    # 現状，labelpos="N"の場合のみ対応

    # 3. 隅にテキストを配置してこれらで自動調節をする
    for x in refvec_corners_for_dummytext[0]:
        for y in refvec_corners_for_dummytext[1]:
            ax.text(
                s="Dummy",
                x=x,
                y=y,
                horizontalalignment="center",
                verticalalignment="center",
                alpha=0.0,
                gid="You can delete this dummy text.",
            )


def plot_vector(
    fig: Figure,
    ax: Axes,
    snap_time_ms: int,
    is_plot_reference_vector: bool,
    group_id_prefix: str,
    group_index: int,
    par_x: NDArray[np.float64],
    par_y: NDArray[np.float64],
    refvec_corners_for_dummytext: List[List[float]],
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_] | None = None,
) -> None:
    par_u, par_v = load_par_data_masked_by_plot_region(
        pardata_filepath_str_time_replaced_by_xxxxx=PLOT_VECTOR_PARAMS.data_file_path,
        snap_time_ms=snap_time_ms,
        mask_array=mask_array,
        usecols=(
            PLOT_VECTOR_PARAMS.col_index_vectorx,
            PLOT_VECTOR_PARAMS.col_index_vectory,
        ),
        mask_array_by_group=mask_array_by_group,
    )

    # scale_units="x"で軸の長さが基準
    # scale=10で，軸単位で0.1の長さが大きさ1のベクトルの長さに対応する
    original_scale = 10 / (IN_PARAMS.xlim_max - IN_PARAMS.xlim_min)
    scale = original_scale / PLOT_VECTOR_PARAMS.scaler_length_vector
    # 5000はチューニング定数
    width = original_scale / 5000 * PLOT_VECTOR_PARAMS.scaler_width_vector

    groupingid = IN_PARAMS.grouping_id
    # ベクトルをプロット
    q = ax.quiver(
        par_x,
        par_y,
        par_u,
        par_v,
        scale=scale,
        scale_units="x",
        width=width,
        headlength=PLOT_VECTOR_PARAMS.headlength_vector,
        headaxislength=PLOT_VECTOR_PARAMS.headaxislength_vector,
        headwidth=PLOT_VECTOR_PARAMS.headwidth_vector,
        gid=f"{group_id_prefix}_{PLOT_GROUP_IDX_PARAMS[groupingid][group_index].label}",
        zorder=PLOT_GROUP_IDX_PARAMS[groupingid][group_index].vector_zorder,
    )

    if is_plot_reference_vector:
        # reference vectorのプロット
        qk = ax.quiverkey(
            Q=q,
            X=PLOT_VECTOR_PARAMS.reference_vector_pos_x,
            Y=PLOT_VECTOR_PARAMS.reference_vector_pos_y,
            U=PLOT_VECTOR_PARAMS.length_reference_vector,
            label=PLOT_VECTOR_PARAMS.reference_vector_text.replace(
                "xxxxx",
                f"{PLOT_VECTOR_PARAMS.length_reference_vector:{PLOT_VECTOR_PARAMS.strformatter_reference_vector_text}}",
            ),
            fontproperties={"size": IN_PARAMS.reference_vector_font_size},
            labelsep=PLOT_VECTOR_PARAMS.reference_vector_labelpad,
            labelpos="N",
            coordinates="data",
        )

        # 初回のみreference vector表示用のdummy textの座標を取得
        if snap_time_ms == IN_PARAMS.snap_start_time_ms:
            update_refvec_corners(
                refvec_corners_for_dummytext=refvec_corners_for_dummytext,
                fig=fig,
                ax=ax,
                qk=qk,
            )

        # reference vector表示用のdummy textをプロット
        plot_transparent_dummytext_for_reference_vector_display(
            ax=ax,
            refvec_corners_for_dummytext=refvec_corners_for_dummytext,
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
        labelbottom=IN_PARAMS.is_plot_ticks_bottom,
        labeltop=IN_PARAMS.is_plot_ticks_top,
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
        labelleft=IN_PARAMS.is_plot_ticks_left,
        labelright=IN_PARAMS.is_plot_ticks_right,
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
        horizontalalignment=IN_PARAMS.ylabel_horizontalalignment,
        fontsize=IN_PARAMS.ylabel_font_size,
        gid="y_title_text",
    )

    if not IN_PARAMS.is_horizontal_ylabel:
        tmp.set_rotation(90.0)

    return


def set_ax_time_text(ax: Axes, snap_time_ms: int) -> None:
    ax.text(
        s=IN_PARAMS.time_text.replace(
            "xxxxx",
            f"{snap_time_ms/IN_PARAMS.scaler_s_to_ms:{IN_PARAMS.strformatter_timetext}}",
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
    groupingid = IN_PARAMS.grouping_id
    change_contour_color = PLOT_GROUP_IDX_PARAMS[groupingid][group_index].contour_color
    change_contour_alpha = PLOT_GROUP_IDX_PARAMS[groupingid][group_index].contour_alpha

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
    snap_time_ms: int,
    plot_name: str,
    mask_array: NDArray[np.bool_],
    mask_array_by_group: NDArray[np.bool_],
) -> NDArray[np.float64]:
    par_color = get_facecolor_by_physics_contour(
        snap_time_ms=snap_time_ms,
        plot_name=plot_name,
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
            c=PLOT_GROUP_IDX_PARAMS[IN_PARAMS.grouping_id][group_index].group_color,
            alpha=PLOT_GROUP_IDX_PARAMS[IN_PARAMS.grouping_id][group_index].group_alpha,
        ),
    )


def get_cur_is_plot_vector(
    plot_name: str, is_group_plot: bool, group_index: int
) -> bool:
    if is_group_plot:
        return PLOT_GROUP_IDX_PARAMS[IN_PARAMS.grouping_id][
            group_index
        ].group_is_plot_vector

    else:
        return (
            PLOT_GROUP_IDX_PARAMS[IN_PARAMS.grouping_id][
                group_index
            ].contour_is_plot_vector
            and PLOT_CONTOUR_PARAMS[plot_name].is_plot_vector
        )


def make_snap_each_snap_time(
    fig: Figure,
    ax: Axes,
    snap_time_ms: int,
    save_dir_snap_path: Path,
    plot_name: str,
    is_group_plot: bool,
    refvec_corners_for_dummytext: List[List[float]],
) -> None:
    set_ax_lim(ax=ax)

    if IN_PARAMS.is_plot_hypervisor2:
        set_ax_xticks(ax=ax)
        set_ax_yticks(ax=ax)
    else:
        ax.axis("off")

    if IN_PARAMS.is_plot_hypervisor2 and IN_PARAMS.is_plot_xlabel_text:
        set_xlabel(ax=ax)
    if IN_PARAMS.is_plot_hypervisor2 and IN_PARAMS.is_plot_ylabel_text:
        set_ylabel(ax=ax)

    if IN_PARAMS.is_plot_hypervisor2 and IN_PARAMS.is_plot_time_text:
        set_ax_time_text(ax=ax, snap_time_ms=snap_time_ms)

    #  初回だけキャンバス更新が必要（scatterの大きさを揃えるため）
    if snap_time_ms == IN_PARAMS.snap_start_time_ms:
        fig.canvas.draw()

    mask_array = get_mask_array_by_plot_region(snap_time_ms=snap_time_ms)

    par_group_index = get_group_index_array(
        snap_time_ms=snap_time_ms, mask_array=mask_array
    )

    particle_group_id_prefix = "particle"
    vector_group_id_prefix = "vector"

    is_plot_reference_vector = PLOT_VECTOR_PARAMS.is_plot_reference_vector

    for group_index in np.unique(par_group_index)[::-1]:
        mask_array_by_group: NDArray[np.bool_] = par_group_index == group_index

        # x, y, disaの取得
        par_x, par_y, par_disa = load_par_data_masked_by_plot_region(
            pardata_filepath_str_time_replaced_by_xxxxx=IN_PARAMS.xydisa_file_path,
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
                snap_time_ms=snap_time_ms,
                plot_name=plot_name,
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
        if get_cur_is_plot_vector(
            plot_name=plot_name, is_group_plot=is_group_plot, group_index=group_index
        ):
            plot_vector(
                fig=fig,
                ax=ax,
                snap_time_ms=snap_time_ms,
                is_plot_reference_vector=is_plot_reference_vector,
                par_x=par_x,
                par_y=par_y,
                group_id_prefix=vector_group_id_prefix,
                group_index=group_index,
                refvec_corners_for_dummytext=refvec_corners_for_dummytext,
                mask_array=mask_array,
                mask_array_by_group=mask_array_by_group,
            )

            # 最初のみreference vectorをプロットする
            is_plot_reference_vector = False

    # 以下，画像の保存処理
    save_file_name_without_extension = (
        f"snap{snap_time_ms:0{IN_PARAMS.num_x_in_pathstr}}_{plot_name}"
    )
    save_file_path = save_dir_snap_path / Path(
        f"{save_file_name_without_extension}.{IN_PARAMS.snap_extension}"
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
    save_dir_snap_path = save_dir_sub_path / "snap_shot"
    save_dir_snap_path.mkdir(exist_ok=True, parents=True)

    scaler_cm_to_inch = 1 / 2.54
    disired_fig_width = IN_PARAMS.fig_horizontal_cm * scaler_cm_to_inch

    print(
        "fig_horizontal（ピクセル単位） :",
        int(disired_fig_width * IN_PARAMS.snap_dpi),
    )
    # 縦方向が見切れないよう十分大きく取る
    fig = plt.figure(
        figsize=(disired_fig_width, 5 * disired_fig_width),
        dpi=IN_PARAMS.snap_dpi,
    )

    # widthはちょうど指定した大きさに近づくようチューニングする
    ax = fig.add_axes(
        ((0, 0, 1 * IN_PARAMS.tooning_for_fig, 1)),
        aspect="equal",
    )

    if IN_PARAMS.is_plot_colorbar and (not is_group_plot):
        plot_colorbar(fig=fig, ax=ax, plot_name=plot_name)

    # reference vectorを表示するための内部処理用の変数の初期化
    refvec_corners_for_dummytext = [[0.0, 0.0], [0.0, 0.0]]

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
                refvec_corners_for_dummytext=refvec_corners_for_dummytext,
            )
            print(
                f"{snap_time_ms/IN_PARAMS.scaler_s_to_ms:.03f} s {plot_name} plot finished"
            )
        except FileNotFoundError as e:
            print(e)
            print(
                f"{snap_time_ms/IN_PARAMS.scaler_s_to_ms:.03f} s時点の計算データがありません．{plot_name}におけるスナップショット作成を終了します．\n"
            )
            break

    print(
        f"{snap_time_ms/IN_PARAMS.scaler_s_to_ms:.03f} s {plot_name} all make snapshot finished\n"
    )
    plt.close()
    return


def make_animation_from_snap(
    anim_time_array_ms: NDArray[np.int64], save_dir_sub_path: Path, plot_name: str
) -> None:
    save_dir_animation_path = save_dir_sub_path / "animation"
    save_dir_animation_path.mkdir(exist_ok=True)

    # 連番でない画像の読み込みに対応させるための準備
    for_ffmpeg = []
    for anim_time_ms in anim_time_array_ms:
        cur_snap_path = (
            save_dir_sub_path
            / "snap_shot"
            / f"snap{anim_time_ms:0{IN_PARAMS.num_x_in_pathstr}}_{plot_name}.jpeg"
        )
        # アニメーション作成で使うsnapが存在するかの確認
        if not cur_snap_path.exists():
            continue
        for_ffmpeg.append(
            f"file 'snap{anim_time_ms:0{IN_PARAMS.num_x_in_pathstr}}_{plot_name}.jpeg'"
        )

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


def execute_plot_all(is_group_plot: bool) -> None:
    # スナップやアニメーションを保存するディレクトリ名
    save_dir_path: Path = Path(__file__).parent / IN_PARAMS.save_dir_name

    # スナップショットを出力する時間[ms]のarray
    snap_time_array_ms: NDArray[np.int64] = np.arange(
        IN_PARAMS.snap_start_time_ms,
        IN_PARAMS.snap_end_time_ms + IN_PARAMS.snap_timestep_ms,
        IN_PARAMS.snap_timestep_ms,
    )

    # アニメーション作成に使用するスナップショットの時間[ms]のarray
    anim_time_array_ms: NDArray[np.int64] = np.arange(
        IN_PARAMS.anim_start_time_ms,
        IN_PARAMS.anim_end_time_ms + IN_PARAMS.anim_timestep_ms,
        IN_PARAMS.anim_timestep_ms,
    )

    plot_order_list = (
        PLOT_GROUP_IDX_PARAMS.keys() if is_group_plot else PLOT_CONTOUR_PARAMS.keys()
    )

    for plot_name in plot_order_list:
        if plot_name == "NOT_PLOT_BELOW":
            break

        save_dir_sub_path = save_dir_path / plot_name

        if IN_PARAMS.is_make_snap:
            make_snap_all_snap_time(
                snap_time_array_ms=snap_time_array_ms,
                save_dir_sub_path=save_dir_sub_path,
                plot_name=plot_name,
                is_group_plot=is_group_plot,
            )

        if IN_PARAMS.is_make_anim:
            make_animation_from_snap(
                anim_time_array_ms=anim_time_array_ms,
                save_dir_sub_path=save_dir_sub_path,
                plot_name=plot_name,
            )

    return


# ---グローバル変数群---
# contour, vector, group以外のすべてのyamlの設定を格納
IN_PARAMS = construct_input_parameters_dataclass()

# contourの設定を格納．plot_order_list_contourの「plot_name -> contour内の設定」の辞書
PLOT_CONTOUR_PARAMS = construct_dict_of_contour_dataclass()

# vectorの設定を格納（vector_idで指定したもの）
PLOT_VECTOR_PARAMS = construct_vector_dataclass()

# groupの設定を格納1．plot_order_list_groupの「group_name -> config -> そのconfig内の設定」の辞書の辞書
PLOT_GROUP_CONFIG_PARAMS = construct_dict_of_group_config_dataclass()

# groupの設定を格納2．plot_order_list_groupの「group_name -> idx -> そのidx内の設定」の辞書の辞書
PLOT_GROUP_IDX_PARAMS = construct_dict_of_group_idx_dataclass()
# ---グローバル変数群---


def main() -> None:
    print(f"plot execute by {sys.argv[1]}\n")

    set_mplparams_init()

    # contourプロット
    execute_plot_all(is_group_plot=False)

    # groupプロット
    execute_plot_all(is_group_plot=True)

    print("all finished")

    return


if __name__ == "__main__":
    print(mpl.matplotlib_fname())
    main()
