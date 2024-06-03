import datetime
import os.path

import streamlit

from example_pipeline import run_pipeline_on_image_directory, example_pipeline
from model.fnet.model_analysis.analyze_trained_models import split_name

streamlit.title('i4SQ Demo')
streamlit.write('This is a demo of the i4SQ tool within the context of the i4Reader toolbox.')
streamlit.write('The i4SQ tool is a tool for the automatic detection of vertebral fractures in CT images.')

uploaded_files = streamlit.file_uploader('Upload a CT image',
                                         type=['dcm', 'nii.gz'],
                                         accept_multiple_files=True)

if uploaded_files:
    # patient_id = datetime.datetime.now().strftime('20240530_190948')
    patient_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    img_dir = os.path.join('uploads', patient_id)
    os.makedirs(img_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        filename = uploaded_file.name
        with open(os.path.join(img_dir, filename), 'wb') as f:
            f.write(bytes_data)
        # streamlit.write("filename:", uploaded_file.name)
        # streamlit.write(len(bytes_data), "bytes")
    p = example_pipeline()
    img = run_pipeline_on_image_directory(patient_id, img_dir, p)
    if img:
        streamlit.write('# Results:')
        model_outputs = img['tool_outputs'][p.fnet.name()]['ifo']
        names = img['tool_outputs'][p.fnet.name()]['names']
        for name, model_output in zip(names, model_outputs):
            p, v = split_name(name)
            # streamlit.write('##', v)
            # streamlit.write('x:', img.parts[v].position[2])
            # streamlit.write('y:', img.parts[v].position[1])
            # streamlit.write('z:', img.parts[v].position[0])
            assert len(model_output) == 1, model_output
            model_output = model_output[0]
            highlight = '' if model_output < 0.5 else '**'
            streamlit.markdown(f'{highlight}{v}: {model_output:.2%} fractured{highlight}')

