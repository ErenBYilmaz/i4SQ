from typing import Dict, Any, Tuple

import SimpleITK
import numpy
import pydicom
from matplotlib import pyplot

from image_types import is_dcm_file, is_mhd_file
from load_data import X, Y, Z
from load_data.ct_metadata import try_convert_to_number, attr_dir

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


def metadata(ct_file: str, patient_id: str, ignore=None) -> Dict:
    if is_dcm_file(ct_file):
        image_attrs = attr_dir(pydicom.dcmread(ct_file), ignore=ignore)

        # sometimes the spacing is missing, then just assume slice thickness to be the spacing
        if 'SpacingBetweenSlices' not in image_attrs:
            image_attrs['SpacingBetweenSlices'] = image_attrs['SliceThickness']
    else:
        assert is_mhd_file(ct_file)
        with open(ct_file, 'r') as mhd_file:
            mhd_content = mhd_file.read()
        image_attrs = {
            k: v if ' ' not in v else v.split(' ')
            for line in mhd_content.splitlines()
            for k, v in [line.split(' = ')]
        }
        try_convert_to_number(image_attrs)
        image_attrs['SpacingBetweenSlices'] = image_attrs['ElementSpacing'][2]

    return {
        **image_attrs,
        'patient_number': (patient_id),
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
