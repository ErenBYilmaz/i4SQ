import os

import cachetools
import streamlit

import hiwi
from dcm_to_nifti import dcm_to_nifti, metadata, read_spacing_from_nii
from lib.util import listdir_fullpath
from model.fnet.fnet import FNet
from model.hnet_fnet.hnet_fnet import HNetFNetPipeline
from model.pbl.pbl import PBL

PBL_MODEL_PATH = r'./models/pbl/model_2020_01_23.pbl'
FNET_MODELS_PATH = r'./models/fnet'


@cachetools.cached(cache=cachetools.LRUCache(maxsize=1))
def pipeline() -> HNetFNetPipeline:
    some_fnet_model = [p for p in listdir_fullpath(FNET_MODELS_PATH) if p.endswith('.h5')][0]
    return HNetFNetPipeline(
        hnet=PBL(
            model_exe_path=PBL_MODEL_PATH,
            model_version=8,
            results_dir='uploads/results',
        ),
        fnet=FNet(
            model_path=some_fnet_model,
        )
    )


def run_pipeline_on_image_directory(patient_id, img_dir):
    filenames = listdir_fullpath(img_dir)
    p = pipeline()
    file_extensions_present = set([os.path.splitext(f)[1] for f in filenames])
    if len(file_extensions_present) > 1:
        streamlit.write('Please upload files of the same type')
    elif len(file_extensions_present) == 0:
        streamlit.write('Please upload at least one file')
    else:
        ext = file_extensions_present.pop()
        if ext == '.nii.gz':
            if len(filenames) > 1:
                streamlit.write('Please upload only one nii.gz file at a time')
            nii_file_path = os.path.join(img_dir, filenames[0])
        else:
            assert ext == '.dcm'
            nii_file_path = os.path.join('uploads', 'results', f'{patient_id}.nii.gz')
            dcm_to_nifti(ct_dir=img_dir,
                         image_metadata=metadata(filenames[0], ignore=['per_vertebra_annotations', 'crf', '3dcpm'], patient_id=patient_id),
                         nifti_path=nii_file_path,
                         patient_id=patient_id)
        assert os.path.isfile(nii_file_path)
        print('Processing', nii_file_path)
        hiwi_image = hiwi.Image(path=nii_file_path)
        hiwi_image['spacing'] = read_spacing_from_nii(nii_file_path)
        hiwi_image['base_dcm_path'] = os.path.dirname(nii_file_path)
        hiwi_image['patient_id'] = nii_file_path
        hiwi_image.objects = [hiwi.Object()]

        p.predict_on_single_image(hiwi_image)


if __name__ == '__main__':
    run_pipeline_on_image_directory('20240530_190948', 'uploads/20240530_190948', )
