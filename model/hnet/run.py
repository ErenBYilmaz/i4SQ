import os

import hiwi
from ct_dirs import RAW_DATA_PATH, PREPROCESSED_PATH
from lib.main_wrapper import main_wrapper
from model.hnet.hnet import HNet, hnet_by_version
from wrapper.config import RESULTS_BASE


@main_wrapper
def main():
    hnet = HNet(model_exe_path=hnet_by_version(7).exe_path,
                results_dir=f'{RESULTS_BASE}/philips_round_{7}')
    print(f'Applying {hnet.name()} ...')
    dirname = os.path.join(PREPROCESSED_PATH, 'nifti', 'Skelett')
    for nii_filename in os.listdir(dirname):
        img = hiwi.Image()
        img.path = os.path.abspath(os.path.join(PREPROCESSED_PATH, 'nifti', 'Skelett', nii_filename))
        input_file = str(img.path)
        if not os.path.isfile(input_file) or not input_file.endswith('.nii.gz'):
            continue
        hnet.predict_on_single_image(img)


if __name__ == '__main__':
    main()
