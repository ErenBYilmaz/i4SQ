import json
import os
import re
from subprocess import CalledProcessError
from typing import List, Dict, Any

import SimpleITK
import numpy

from hiwi import Image
from lib.call_tool import call_tool
from model.hnet.hnet import HNet


def parse_pbl_output(pbl_output):
    # pbl_output = re.search(r'Output of `.*'
    #                        + escaped_pbl_model_path
    #                        + r' .*'
    #                        + escaped_pid
    #                        + r'.*\r?\n```\r?\n([\s\S]*?\r?\n)```\r?\n',
    #                        scan_result).group(1)
    pbl_output_vertebrae: List[Dict[str, Any]] = []
    for line in pbl_output.splitlines():
        # lines are expected to have the format `<vertebra>: <x> <y> <z>`
        r = re.search(r'^(.+?):\s*?(-?[0-9.]+?)?\s+(-?[0-9.]+?)\s+(-?[0-9.]+?)$', line)
        if r is None:
            continue  # no match
        vertebra_name: str
        vertebra_name, x, y, z = r.groups()
        if 'C' in vertebra_name:
            continue  # model was not trained on this kind of vertebrae
        pbl_output_vertebrae.append({
            'name': vertebra_name,
            'x': round(float(x)),  # physical point
            'y': round(float(y)),
            'z': round(float(z)),
        })
    return pbl_output_vertebrae


class PBL(HNet):
    def nii_to_pbl_spacing(self, nii_file_path: str, spaced_path: str):
        s_image = SimpleITK.ReadImage(nii_file_path)
        pbl_required_spacing = (1.5, 1.5, 1.5)
        scaling = numpy.array(s_image.GetSpacing()) / 1.5
        r = SimpleITK.ResampleImageFilter()
        r.SetInterpolator(SimpleITK.sitkBSpline)
        r.SetOutputSpacing(pbl_required_spacing)
        r.SetOutputDirection(s_image.GetDirection())
        r.SetSize(numpy.ceil(s_image.GetSize() * scaling).astype(int).tolist())
        physical_origin = s_image.TransformContinuousIndexToPhysicalPoint((0 + .5) / scaling - .5)
        r.SetOutputOrigin(physical_origin)
        s_image = r.Execute(s_image)
        SimpleITK.WriteImage(s_image, spaced_path)

    def predict_on_single_image(self, img: Image) -> dict:
        pbl_model_path = os.path.abspath(self.model_exe_path)
        input_path = str(img.path)
        spaced_path = input_path.replace('.nii.gz', '_spaced.nii.gz')
        self.nii_to_pbl_spacing(input_path, spaced_path)

        input_path = os.path.abspath(spaced_path)
        command = ["python", "-m", "pbl", "test", "--output-dir", "log/pbl-outputs", pbl_model_path, input_path]
        nii_filename = os.path.basename(input_path)
        result_name = self.result_name_for_input(nii_filename)
        tool = self.model_exe_path

        ext = '.nii.gz'
        assert input_path.endswith(ext)
        json_result_path = os.path.join(os.path.abspath(result_name) + '_spine_fine.json')

        try:
            if os.path.isfile(json_result_path):
                print(json_result_path, 'already exists. Using the existing file.')
            else:
                tool_output = call_tool(command, force_cpu=True, cwd=os.path.abspath(os.path.dirname(os.path.dirname(tool))))
                pbl_output_vertebrae = parse_pbl_output(tool_output)
                print(pbl_output_vertebrae)
                raise NotImplementedError('TODO write json in same format as hNet')
        except CalledProcessError:
            print(CalledProcessError)
            print('Ignoring a failed process call and dumping empty json file..')
            with open(json_result_path, 'w') as f:
                json.dump({}, f)
        assert os.path.isfile(json_result_path), json_result_path
        mps_path = self.json_file_to_mps_file(json_result_path)

        outputs = self.predictions_from_json_file(json_result_path)

        assert os.path.isfile(mps_path)
        outputs['_mps_path'] = mps_path

        return outputs
