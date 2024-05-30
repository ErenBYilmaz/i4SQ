import os.path
import shutil
import datetime

FRACTURES_PATH = r'D:\eren\Software\Fractures'
SPINETEAM_DATA_PATH = r'C:\Users\sukin692\PycharmProjects\spineteam-data'


def main():
    copy_from_dirs = [FRACTURES_PATH, SPINETEAM_DATA_PATH]
    for p in copy_from_dirs:
        assert os.path.isdir(p), p
    copy_to_dir = os.path.dirname(__file__)

    copy_paths_relative = [
        'model/fnet',
        'model/ensemble',
        'model/hnet',
        'model/pbl',
        'model/hnet_fnet',
        'hiwi',
        'load_data',
        'tasks.py',
        'image_types.py',
        'lib',
        'data/plot_patches.py',
        'data/vertebrae.py',
    ]

    for p in copy_paths_relative:
        in_paths = [os.path.join(from_path, p) for from_path in copy_from_dirs]
        in_paths = [d for d in in_paths if os.path.exists(d)]
        assert len(in_paths) == 1, (in_paths, p)
        in_dir = in_paths[0]
        out_name = prepare_relative_output_dir(copy_to_dir, p)
        if os.path.exists(out_name):
            bak_name = prepare_relative_output_dir(copy_to_dir, f'old/{p}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
            shutil.move(out_name, bak_name)
        copy_file_or_dir(in_dir, out_name)


def copy_file_or_dir(in_dir, out_name):
    assert not os.path.exists(out_name)
    if os.path.isdir(in_dir):
        shutil.copytree(in_dir, out_name)
    else:
        assert os.path.isfile(in_dir)
        shutil.copy(in_dir, out_name)


def prepare_relative_output_dir(copy_to_dir, p) -> str:
    out_name: str = os.path.join(copy_to_dir, p)
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    return out_name


if __name__ == '__main__':
    main()
