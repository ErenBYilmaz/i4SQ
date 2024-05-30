import os
from typing import Dict, Any, Tuple

import SimpleITK
import numpy
from matplotlib import pyplot

from load_data import X, Y, Z
from load_data.ct_metadata import try_convert_to_number

Size = Tuple[X, Y, Z]
Spacing = Tuple[X, Y, Z]


def get_orientation(metadata: Dict[str, Any]):
    return metadata['PatientPosition']


def read_spacing_from_nii(filename):
    import SimpleITK as sitk
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    spacing = reader.GetSpacing()
    return spacing
def metadata(ct_file: str, patient_id:str,  ignore=None) -> Dict:
    if is_dcm_file(ct_file):
        image_attrs = attr_dir(pydicom.dcmread(ct_file), ignore=ignore)

        # sometimes the spacing is missing, then just assume slice thickness to be the spacing
        if 'SpacingBetweenSlices' not in image_attrs:
            image_attrs['SpacingBetweenSlices'] = image_attrs['SliceThickness']

        df = excel_panda(EXCEL_FILE, header=1)
        df = df[df['Patients Name'] == int(patient_id)]
        genant_0 = df[df['SQ Score'] == 0].shape[0]
        genant_1 = df[df['SQ Score'] == 1].shape[0]
        genant_2 = df[df['SQ Score'] == 2].shape[0]
        genant_3 = df[df['SQ Score'] == 3].shape[0]
    else:
        assert is_mhd_file(ct_file)
        df = csv_data(score_file,
                      patient_number=(patient_id),
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
                                                  patient_number=(patient_id),
                                                  header=1)
            per_vertebra_annotations = {
                v['Label']: v
                for v in per_vertebra_annotations.to_dict(orient='index').values()
            }
            for vertebra_name in per_vertebra_annotations:
                if ((patient_id), vertebra_name) in actually_osteoporotic:
                    assert 'Differential Diagnosis Category' in per_vertebra_annotations[vertebra_name]
                    assert per_vertebra_annotations[vertebra_name]['SQ Score'] >= 1  # if this triggers the radiologist probably made a mistake
                    per_vertebra_annotations[vertebra_name]['Differential Diagnosis Category'] = 'Osteoporotic Fracture'
        except PatientNotFoundError:
            per_vertebra_annotations = csv_data(score_file,
                                                patient_number=(patient_id),
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
                if patient_id in use_other_tiff_file:
                    filename_to_search = use_other_tiff_file[patient_id].lower()
                else:
                    if 'Filename' not in annotation:
                        continue
                    filename_to_search = annotation['Filename'].lower()
                for path, directory, filenames in os.walk(os.path.join(TIFF_DIR, str(patient_id))):
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
            with open(os.path.join(CRF_DIR, str(patient_id) + '.json')) as json_file:
                crf = json.load(json_file)
                crf = remove_other_methods(crf, 'Loc. + CRF (weighted, latent scale)')
        except FileNotFoundError:
            # print('WARNING: No crf for ' + str(patient_number))
            crf = None

    if '3dcpm' in ignore:
        three_dcpm = None
    else:
        try:
            with open(os.path.join(THREE_DCPM_DIR, str(patient_id) + '.json')) as json_file:
                three_dcpm = json.load(json_file)
                three_dcpm = remove_other_methods(three_dcpm, '3_refined')
        except FileNotFoundError:
            # print('WARNING: No three_dcpm for ' + str(patient_number))
            three_dcpm = None

    return {
        **image_attrs,
        'patient_number': (patient_id),
        'crf': crf,
        '3dcpm': three_dcpm,
        'per_vertebra_annotations': per_vertebra_annotations,
        'sum_score': sum_score,
        'genant_0': genant_0,
        'genant_1': genant_1,
        'genant_2': genant_2,
        'genant_3': genant_3,
    }

def dcm_to_nifti(ct_dir: str,
                 image_metadata: dict,
                 nifti_path: str,
                 patient_id: str) -> Tuple[Size, Spacing]:
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
    reader.SetFileNames(dicom_names)
    s_image: SimpleITK.Image = reader.Execute()
    orientation = get_orientation(image_metadata)
    if orientation == 'FFS':
        old_data = SimpleITK.GetArrayFromImage(s_image)
        old_direction = s_image.GetDirection()
        flip_image_filter = SimpleITK.FlipImageFilter()
        flip_image_filter.SetFlipAxes([False, False, True])
        s_image = flip_image_filter.Execute(s_image)
        s_image.SetDirection(old_direction)  # This flipping was done to flip the voxels only, the image was wrongly oriented w.r.t. the direction annotation
        assert not numpy.array_equal(old_data, SimpleITK.GetArrayFromImage(s_image))
    elif orientation == 'HFS':
        pass
        # flip_image_filter = SimpleITK.FlipImageFilter()
        # flip_image_filter.SetFlipAxes([False, False, True])
        # print('Not flipping image for patient ' + patient_id)
        # flipped_image = flip_image_filter.Execute(s_image)
        # flipped_image_array = SimpleITK.GetArrayViewFromImage(flipped_image)
        # assert not numpy.array_equal(flipped_image_array, SimpleITK.GetArrayFromImage(s_image))
        # pyplot.imsave(nifti_path + '_sagittal_view_if_it_would_have_been_flipped.png', flipped_image_array[:, :, flipped_image_array.shape[2] // 2])
    else:
        raise NotImplementedError(f'Unknown PatientPosition (0018,5100) value `{orientation}` in {patient_id}. '
                                  f'Only FFS and HFS are known. '
                                  f'If your scanner created  a CT image with a different orientation you have multiple options: '
                                  f'1. File a bug report and wait until we fix it. '
                                  f'2. Transform your dcm files such that the patient actually is in FFS or HFS position'
                                  f' and then set the annotation accordingly.')
    image_array = SimpleITK.GetArrayFromImage(s_image)
    pyplot.imsave(nifti_path + '_sagittal_view.png', image_array[:, :, image_array.shape[2] // 2], cmap='gray')
    assert s_image.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    SimpleITK.WriteImage(s_image, nifti_path)
    size, spacing = s_image.GetSize(), s_image.GetSpacing()
    return size, spacing
