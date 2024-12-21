from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def get_floatdata_from_input(search_string: str) -> float:
    with open(Path(__file__).parents[2] / "input.dat", mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    dataline_split = [
        line.strip().split() for line in lines if search_string in line.strip().split()
    ]
    assert len(dataline_split) == 1

    res_data = float(dataline_split[0][2])

    return res_data


def get_dict_of_snaptimems_to_startrowidx_and_nrow(
    outputdat_path: Path,
    timeInterval_ms: int,
    timeStart_ms: int,
) -> Dict[int, tuple[int, int]]:
    convert_log_list: List[List[str | float]] = []

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
            converted_inf = 9.1234 * 10**99
            print(
                f"素データの数値が大きすぎて，指数部を表すEが存在しないパターンが検出されました（{x}）．符号付きE+99程度の大きさの数値に置き換えます．"
            )
            # Eがない場合には特殊ルール適用
            # 必ず+を先にすること，先頭の-対策
            if "+" in x:
                convert_log_list.append([x, converted_inf])
                return converted_inf
            else:
                convert_log_list.append([x, -converted_inf])
                return -converted_inf

    snaptimems_to_startrowidx_nrow = dict()

    nrow_header = 4

    cur_row = 0
    cur_time_ms = timeStart_ms

    save_dir_path = Path(__file__).parents[2] / "plot_particles_mode_p_log"

    for_mode_p_tmp = []

    with pd.read_table(
        outputdat_path,
        iterator=True,
        header=None,
        comment="#",
        escapechar="#",
        usecols=[0],
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

                # headerの情報を抽出＆型変換
                time_s = float(header.iat[0, 0].split()[0])
                timestep = int(header.iat[1, 0].split()[0])
                # particle + dem
                number_of_particles = int(header.iat[2, 0].split()[0])
                n0 = float(header.iat[2, 0].split()[1])

                # print debug
                print(f"{header=}")
                print(f"{time_s=}, {number_of_particles=}")

                # データ部分を読み進める
                print(np.array(reader.get_chunk(number_of_particles)))

                # snaptimems_to_startrowidx_nrowを更新
                startrow_idx = cur_row + nrow_header
                snaptimems_to_startrowidx_nrow[cur_time_ms] = (
                    startrow_idx,
                    number_of_particles,
                )

                # for_mode_p.datに情報を保存するために
                for_mode_p_tmp.append(
                    [cur_time_ms, startrow_idx, number_of_particles, timestep, n0]
                )

                # cur情報を更新．次の時刻の処理へ
                cur_row += nrow_header + number_of_particles + 1  # +1は空白行分
                cur_time_ms += timeInterval_ms

            except pd.errors.EmptyDataError:
                print("for_mode_p.pyの初期化処理完了")

                save_dir_path.mkdir(exist_ok=True)

                # for_mode_p.datを保存
                np.savetxt(
                    save_dir_path / "for_mode_p.dat",
                    for_mode_p_tmp,
                    fmt=["%d", "%d", "%d", "%d", "%f"],
                    header="time(ms), startrow_idx, number_of_particle, timestep, n0",
                )

                # convert_log.datを保存
                np.savetxt(
                    save_dir_path / "convert_log.dat",
                    convert_log_list,
                    header="original_data, converted_data",
                )

                return snaptimems_to_startrowidx_nrow

        else:
            raise ValueError("for_mode_p.pyの初期化処理の際に問題が発生しました")


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

        # for_mode_p.datの前処理ファイルがあればそちらを読み込む
        try:
            tmp = np.loadtxt(
                Path(__file__).parents[2]
                / "plot_particles_mode_p_log"
                / "for_mode_p.dat",
                usecols=(0, 1, 2),
                dtype=np.int_,
            )

            self.snaptimems_to_startrowidx_nrow = {
                tmp[i, 0]: (tmp[i, 1], tmp[i, 2]) for i in range(tmp.shape[0])
            }

        except FileNotFoundError:
            print("output.datの前処理（各スナップの開始行番号等の記録）を実行")
            self.snaptimems_to_startrowidx_nrow = (
                get_dict_of_snaptimems_to_startrowidx_and_nrow(
                    outputdat_path=self.outputdat_path,
                    timeInterval_ms=self.timeInterval_ms,
                    timeStart_ms=self.timeStart_ms,
                )
            )

        print(self.snaptimems_to_startrowidx_nrow)
        print("ForModePの初期化完了")

        return

    def get_original_data(
        self, snap_time_ms: int, usecols: tuple[int, ...] | int, dtype: str
    ) -> NDArray[np.float32 | np.int32]:
        return np.loadtxt(
            self.outputdat_path,
            usecols=usecols,
            skiprows=self.snaptimems_to_startrowidx_nrow[snap_time_ms][0],
            max_rows=self.snaptimems_to_startrowidx_nrow[snap_time_ms][1],
            dtype=dtype,
            encoding="utf-8",
        )
