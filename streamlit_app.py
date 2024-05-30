import datetime
import os.path

import streamlit

streamlit.title('i4SQ Demo')
streamlit.write('This is a demo of the i4SQ tool within the context of the i4Reader toolbox.')
streamlit.write('The i4SQ tool is a tool for the automatic detection of vertebral fractures in CT images.')

uploaded_files = streamlit.file_uploader('Upload a CT image',
                                         type=['dcm', 'nii', 'nii.gz'],
                                         accept_multiple_files=True)
if uploaded_files:
    upload_dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    img_dir = os.path.join('uploads', upload_dt_str)
    os.makedirs(img_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        filename = uploaded_file.name
        with open(os.path.join(img_dir, filename), 'wb') as f:
            f.write(bytes_data)
        streamlit.write("filename:", uploaded_file.name)
        streamlit.write(len(bytes_data), "bytes")