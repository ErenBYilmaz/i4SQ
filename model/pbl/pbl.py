import json
import os
import re
from subprocess import CalledProcessError
from typing import List, Dict, Any

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
    def predict_on_single_image(self, img: Image) -> dict:
        pbl_model_path = self.model_exe_path
        input_path = str(img.path)
        command = ["pbl", "test", "--output-dir", "log/pbl-outputs", pbl_model_path, input_path]
        nii_filename = os.path.basename(input_path)
        result_name = self.result_name_for_input(nii_filename)
        tool = self.model_exe_path
        command = [os.path.abspath(tool), os.path.abspath(input_path), os.path.abspath(result_name)]

        ext = '.nii.gz'
        assert input_path.endswith(ext)
        json_result_path = os.path.join(os.path.abspath(result_name) + '_spine_fine.json')

        try:
            if os.path.isfile(json_result_path):
                print(json_result_path, 'already exists. Using the existing file.')
            else:
                tool_output = call_tool(command, force_cpu=True, cwd=os.path.abspath(os.path.dirname(os.path.dirname(tool))))
                pbl_output_vertebrae = parse_pbl_output(tool_output)
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
