from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def get_floatdata_from_input(search_string: str) -> float:
    with open(Path(__file__).parents[2] / "input.dat", mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    dataline_split = [
        line.strip().split() for line in lines if search_string in line.strip().split()
    ]
    assert len(dataline_split) == 1

    res_data = float(dataline_split[0][2])

    return res_data


def converter_for_floatnotcontainE(x: str) -> np.float32 | float:
    """素データの数値が大きすぎて，0.1234234のように指数部を表すEが存在しない場合のエラー処理

    Args:
        x (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # もし"E"を含む場合は通常通りfloat化（例: "1.23E+02"など）
    try:
        return np.float32(x)
    except:  # noqa: E722
        print(
            "素データの数値が大きすぎて，指数部を表すEが存在しないパターンが検出されました．符号付きE+99程度の大きさの数値に置き換えます．"
        )
        # Eがない場合には特殊ルール適用
        if "-" in x:
            return -9.1234 * 10**99
        elif "+" in x:
            return 9.1234 * 10**99
        else:
            # どちらも含まれない場合はそもそもありえないはずなので止める
            raise ValueError("output.datの読み取り時に問題が発生しました．")


def get_dict_of_snaptimems_to_startrowidx_and_nrow(
    outputdat_path: Path,
    timeInterval_ms: int,
    timeStart_ms: int,
) -> Dict[int, tuple[int, int]]:
    snaptimems_to_startrowidx_nrow = dict()

    nrow_header = 4

    cur_row = 0
    cur_time_ms = timeStart_ms

    with pd.read_table(
        outputdat_path,
        iterator=True,
        comment="#",
        usecols=[0],
        escapechar="#",
        encoding="utf-8",
        sep=r"\s+",
        dtype=np.int32,
    ) as reader:
        for i in range(10000000):
            try:
                # header部分
                header = pd.read_table(
                    outputdat_path,
                    skiprows=cur_row,
                    nrows=nrow_header,
                    escapechar="#",
                    encoding="utf-8",
                    header=None,
                    dtype=str,
                )

                # スナップの時刻と粒子数を抽出＆型変換
                time_s = float(header.iat[0, 0].split()[0])
                number_of_particles = int(header.iat[2, 0].split()[0])

                # print debug
                print(f"{header=}")
                print(f"{time_s=}, {number_of_particles=}")

                # データ部分を読み進める
                reader.get_chunk(number_of_particles)

                # snaptimems_to_startrowidx_nrowを更新
                nrow = nrow_header + number_of_particles
                snaptimems_to_startrowidx_nrow[cur_time_ms] = (cur_row, nrow)

                # cur情報を更新．次の時刻の処理へ
                cur_row += nrow_header + number_of_particles + 1  # +1は空白行分
                cur_time_ms += timeInterval_ms

            except pd.errors.EmptyDataError:
                print("for_mode_p.pyの初期化処理完了")
                return snaptimems_to_startrowidx_nrow

        else:
            raise ValueError("for_mode_p.pyの初期化処理の際に問題が発生しました")


# TODO ファイルがないときの処理
class ForModeP:
    def __init__(self, outputdat_path: Path, scaler_s_to_ms: int) -> None:
        self.outputdat_path = outputdat_path
        self.timeStart_ms = int(
            get_floatdata_from_input(search_string="timeStart") * scaler_s_to_ms
        )
        self.timeInterval_ms = int(
            get_floatdata_from_input(search_string="timeInterval") * scaler_s_to_ms
        )
        self.d0 = get_floatdata_from_input(search_string="d0")
        self.snaptimems_to_startrowidx_nrow = (
            get_dict_of_snaptimems_to_startrowidx_and_nrow(
                outputdat_path=self.outputdat_path,
                timeInterval_ms=self.timeInterval_ms,
                timeStart_ms=self.timeStart_ms,
            )
        )
        return
