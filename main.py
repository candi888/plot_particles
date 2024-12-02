import subprocess
from pathlib import Path


def main() -> None:
    input_yaml_dir_path = Path(__file__).parent / "input_yaml"
    input_yaml_file_list = list(input_yaml_dir_path.glob("*.yaml"))
    input_yaml_file_list.sort()

    for input_yaml_file in input_yaml_file_list:
        cmd = f"python plot.py {input_yaml_file.name}"
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
