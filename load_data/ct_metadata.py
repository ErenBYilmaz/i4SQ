import functools

import SimpleITK

from hiwi import ImageList

from lib.tuned_cache import TunedMemory
import json
import os
from typing import Dict, Any

import SimpleITK as sitk
import pydicom
from PIL import Image, TiffTags

from ct_dirs import CRF_DIR, THREE_DCPM_DIR, NIFTI_OUTPUT_DIR, TIFF_DIR, EXCEL_FILE
from lib.util import dicom_metadata_from_image
from load_data.processing_utils import attr_dir, try_convert_to_number
from load_data.table_data import excel_data, excel_panda, PatientNotFoundError, csv_data
from image_types import is_mhd_file, is_dcm_file

metadata_cache = TunedMemory(location='./.cache/metadata', verbose=0)
LUMBAR_VERTEBRA_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5', ]


class NiftiNotFoundError(FileNotFoundError):
    pass


@metadata_cache.cache
def nifti_spacing(patient_number):
    reader = sitk.ImageFileReader()
    # reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(os.path.join(NIFTI_OUTPUT_DIR, patient_number + '.nii.gz'))
    image: sitk.Image = reader.Execute()
    spacing_in_position = image.GetSpacing()
    return spacing_in_position


def remove_other_methods(data: Dict[str, Dict[str, Any]], keep_method: str):
    data = data.copy()
    for key in data:
        dkm = data[key]['methods']
        assert keep_method in dkm
        for method in list(dkm.keys()):
            if method == keep_method:
                continue
            del dkm[method]
        assert keep_method in dkm
        data[key]['methods'] = dkm
    return data


class MissingTiffError(KeyError):
    pass


def case_by_idx(i: int):
    i = int(i)
    if i == 0:
        return ''
    if i == 1:
        return ' Wedge'
    if i == 2:
        return ' Biconcave'
    if i == 3:
        return ' Crush'
    raise ValueError


def severity(gs: int):
    gs = int(gs)
    if gs == 0:
        return 'Normal'
    if gs == 1:
        return 'Mild'
    if gs == 2:
        return 'Moderate'
    if gs == 3:
        return 'Severe'
    raise ValueError


@functools.lru_cache(maxsize=2000)
@metadata_cache.cache
def vertebra_metadata(patient_id, vertebra_name):
    iml = ImageList.load(os.path.join(DERIVED_PATH, 'spacing_3_1_1', 'clean_patients.iml'))
    for img in iml:
        if str(img.name) == str(patient_id):
            return img.parts[vertebra_name]
    raise KeyError


# @metadata_cache.cache
def vertebra_metadata_check(patient_id, vertebra_name, metadata):
    meta = vertebra_metadata(patient_id, vertebra_name)
    for k, vs in metadata.items():
        if meta[k] in vs:
            return True
    return False


@metadata_cache.cache
def metadata(ct_file: str, ignore=None) -> Dict:
    if ignore is None:
        ignore = ['PixelData', 'pixel_array', 'crf', '3dcpm', 'per_vertebra_annotations']
    patient_number = patient_number_from_long_string(ct_file)
    score_file = os.path.join(*os.path.normpath(ct_file).split('\\')[:-2], 'scores', 'scores.csv')

    if is_dcm_file(ct_file):
        image_attrs = attr_dir(pydicom.dcmread(ct_file), ignore=ignore)

        # sometimes the spacing is missing, then just assume slice thickness to be the spacing
        if 'SpacingBetweenSlices' not in image_attrs:
            image_attrs['SpacingBetweenSlices'] = image_attrs['SliceThickness']

        df = excel_panda(EXCEL_FILE, header=1)
        df = df[df['Patients Name'] == int(patient_number)]
        genant_0 = df[df['SQ Score'] == 0].shape[0]
        genant_1 = df[df['SQ Score'] == 1].shape[0]
        genant_2 = df[df['SQ Score'] == 2].shape[0]
        genant_3 = df[df['SQ Score'] == 3].shape[0]
    else:
        assert is_mhd_file(ct_file)
        df = csv_data(score_file,
                      patient_number=patient_number,
                      header=1)
        with open(ct_file, 'r') as mhd_file:
            mhd_content = mhd_file.read()
        image_attrs = {
            k: v if ' ' not in v else v.split(' ')
            for line in mhd_content.splitlines()
            for k, v in [line.split(' = ')]
        }
        try_convert_to_number(image_attrs)
        image_attrs['SpacingBetweenSlices'] = image_attrs['ElementSpacing'][2]
        genant_0 = sum(1 for vertebra_name in LUMBAR_VERTEBRA_NAMES
                       if df[vertebra_name + ' grade'] == 0)
        genant_1 = sum(1 for vertebra_name in LUMBAR_VERTEBRA_NAMES
                       if df[vertebra_name + ' grade'] == 1)
        genant_2 = sum(1 for vertebra_name in LUMBAR_VERTEBRA_NAMES
                       if df[vertebra_name + ' grade'] == 2)
        genant_3 = sum(1 for vertebra_name in LUMBAR_VERTEBRA_NAMES
                       if df[vertebra_name + ' grade'] == 3)

    if 'per_vertebra_annotations' not in ignore:
        try:
            per_vertebra_annotations = excel_data(EXCEL_FILE,
                                                  patient_number=patient_number,
                                                  header=1)
            per_vertebra_annotations = {
                v['Label']: v
                for v in per_vertebra_annotations.to_dict(orient='index').values()
            }
            for vertebra_name in per_vertebra_annotations:
                if (patient_number, vertebra_name) in actually_osteoporotic:
                    assert 'Differential Diagnosis Category' in per_vertebra_annotations[vertebra_name]
                    assert per_vertebra_annotations[vertebra_name]['SQ Score'] >= 1  # if this triggers the radiologist probably made a mistake
                    per_vertebra_annotations[vertebra_name]['Differential Diagnosis Category'] = 'Osteoporotic Fracture'
        except PatientNotFoundError:
            per_vertebra_annotations = csv_data(score_file,
                                                patient_number=patient_number,
                                                header=1)
            per_vertebra_annotations = {
                v: {
                    'SQ Score': grade,
                    'SQ Type': severity(grade) + case_by_idx(case),
                    'Differential Diagnosis': None,
                    'Differential Diagnosis Category': None,
                    'Contours Adjusted?': None,
                    'Morphometry Adjusted?': None,
                    'Morphometry Viewed?': None,
                    'Height Posterior': None,
                    'Height Anterior': None,
                    'Height Mid': None,
                    'Ratio Wedge': None,
                    'Ratio Biconcave': None,
                    'Ratio Crush': None,
                    'Deformity Wedge': None,
                    'Deformity Biconcave': None,
                    'Deformity Crush': None,
                    'Prevalent Classification Wedge': grade if case == 1 else 0,
                    'Prevalent Classification Biconcave': grade if case == 2 else 0,
                    'Prevalent Classification Crush': grade if case == 3 else 0
                }
                for v in LUMBAR_VERTEBRA_NAMES
                for grade in [int(round(per_vertebra_annotations[v + ' grade']))]
                for case in [int(round(per_vertebra_annotations[v + ' case']))]}

        sum_score = sum(annotation['SQ Score']
                        for annotation in per_vertebra_annotations.values())

        if 'tiff' not in ignore:
            for vertebra, annotation in per_vertebra_annotations.copy().items():
                found = False
                tiff_metadata = None
                if patient_number in use_other_tiff_file:
                    filename_to_search = use_other_tiff_file[patient_number].lower()
                else:
                    if 'Filename' not in annotation:
                        continue
                    filename_to_search = annotation['Filename'].lower()
                for path, directory, filenames in os.walk(os.path.join(TIFF_DIR, str(patient_number))):
                    for filename in [f for f in filenames if f.lower() == filename_to_search]:
                        with Image.open(os.path.join(path, filename)) as img:
                            tiff_metadata = {TiffTags.TAGS[key]: img.tag[key] for key in img.tag.keys()}
                        assert not found
                        found = True
                    #     break
                    # if found:
                    #     break
                if not found:
                    raise MissingTiffError('2D tiff not found: ' + filename_to_search)
                assert tiff_metadata is not None
                per_vertebra_annotations[vertebra]['tiff_metadata'] = tiff_metadata
    else:
        per_vertebra_annotations = None
        sum_score = None

    if 'crf' in ignore:
        crf = None
    else:
        try:
            with open(os.path.join(CRF_DIR, str(patient_number) + '.json')) as json_file:
                crf = json.load(json_file)
                crf = remove_other_methods(crf, 'Loc. + CRF (weighted, latent scale)')
        except FileNotFoundError:
            # print('WARNING: No crf for ' + str(patient_number))
            crf = None

    if '3dcpm' in ignore:
        three_dcpm = None
    else:
        try:
            with open(os.path.join(THREE_DCPM_DIR, str(patient_number) + '.json')) as json_file:
                three_dcpm = json.load(json_file)
                three_dcpm = remove_other_methods(three_dcpm, '3_refined')
        except FileNotFoundError:
            # print('WARNING: No three_dcpm for ' + str(patient_number))
            three_dcpm = None

    return {
        **image_attrs,
        'patient_number': patient_number,
        'crf': crf,
        '3dcpm': three_dcpm,
        'per_vertebra_annotations': per_vertebra_annotations,
        'sum_score': sum_score,
        'genant_0': genant_0,
        'genant_1': genant_1,
        'genant_2': genant_2,
        'genant_3': genant_3,
    }


def metadata_dict_from_image(image: SimpleITK.Image) -> dict:
    if image is None:
        return {}
    result = dicom_metadata_from_image(image)
    result['size'] = image.GetSize()
    result['origin'] = image.GetOrigin()
    result['direction'] = image.GetDirection()
    result['spacing'] = image.GetSpacing()
    result['pixel_type'] = SimpleITK.GetPixelIDValueAsString(image.GetPixelID())
    return result


if __name__ == '__main__':
    # noinspection PyBroadException
    from load_data.image_files_for_ct import image_files_for_ct
    from train_test_split import CLEAN_CT_DIRS, use_other_tiff_file, patient_number_from_long_string, actually_osteoporotic

    one_image_file = image_files_for_ct([CLEAN_CT_DIRS[0]])[CLEAN_CT_DIRS[0]][0]
    m = metadata(one_image_file, ignore=['PixelData', 'pixel_array', 'crf', '3dcpm', 'tiff'])
    print(m['per_vertebra_annotations']['T6'])
