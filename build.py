import os.path

FRACTURES_PATH = r'D:\eren\Software\Fractures'
SPINETEAM_DATA_PATH = r'C:\Users\sukin692\PycharmProjects\spineteam-data'

copy_from_dirs = [FRACTURES_PATH, SPINETEAM_DATA_PATH]
copy_to_dir = os.path.dirname(__file__)

copy_paths_relative = [
    'model/fnet'
    'model/ensemble'
]

def main():
    with open('.gitignore', 'w') as f:
        extra_ignores = ['.gitignore', '__pycache__', '.idea']
        f.writelines(extra_ignores)
        f.writelines(copy_paths_relative)

    for p in copy_paths_relative:
        in_dirs = [os.path.join(from_path, p) for from_path in copy_from_dirs]
        in_dirs = [d for d in in_dirs if os.path.isdir(d)]
        assert len(in_dirs) == 1, in_dirs
        in_dir = in_dirs[0]
        out_name = os.path.join(copy_to_dir, p)
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        shutil.copytree(in_dir, out_name)

if __name__ == '__main__':
    main()