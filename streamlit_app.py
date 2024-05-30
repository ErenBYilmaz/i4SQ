import datetime
import os.path

import streamlit

import example_pipeline
import hiwi
from dcm_to_nifti import dcm_to_nifti, read_spacing_from_nii
from dcm_to_nifti import metadata

streamlit.title('i4SQ Demo')
streamlit.write('This is a demo of the i4SQ tool within the context of the i4Reader toolbox.')
streamlit.write('The i4SQ tool is a tool for the automatic detection of vertebral fractures in CT images.')

uploaded_files = streamlit.file_uploader('Upload a CT image',
                                         type=['dcm', 'nii.gz'],
                                         accept_multiple_files=True)
if uploaded_files:
    patient_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    img_dir = os.path.join('uploads', patient_id)
    os.makedirs(img_dir, exist_ok=True)
    filenames = []
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        filename = uploaded_file.name
        with open(os.path.join(img_dir, filename), 'wb') as f:
            f.write(bytes_data)
        streamlit.write("filename:", uploaded_file.name)
        streamlit.write(len(bytes_data), "bytes")
        filenames.append(filename)
    pipeline = example_pipeline.pipeline()
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
            nii_file_path = os.path.join(img_dir, 'ct_image.nii.gz')
            dcm_to_nifti(ct_dir=img_dir,
                         image_metadata=metadata(filenames[0], ignore=['per_vertebra_annotations', 'crf', '3dcpm']),
                         nifti_path=nii_file_path,
                         patient_id=patient_id)
        assert os.path.isfile(nii_file_path)
        hiwi_image = hiwi.Image(path=nii_file_path)
        hiwi_image['spacing'] = read_spacing_from_nii(nii_file_path)
        hiwi_image['base_dcm_path'] = os.path.dirname(nii_file_path)
        hiwi_image['patient_id'] = nii_file_path
        hiwi_image.objects = [hiwi.Object()]

        pipeline.predict_on_single_image(hiwi_image)
