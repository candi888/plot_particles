import subprocess
from pathlib import Path


def main() -> None:
    input_yaml_dir_path = Path(__file__).parent / "input_yaml"
    input_yaml_file_path_list = list(input_yaml_dir_path.glob("*.yaml"))
    input_yaml_file_path_list.sort()

    for input_yaml_file_path in input_yaml_file_path_list:
        cmd = (
            f"python {str(Path(__file__).parent/"plot.py")} {input_yaml_file_path.name}"
        )
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
