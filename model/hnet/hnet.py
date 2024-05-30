import json
import math
import os
import shutil
import tempfile
from subprocess import CalledProcessError
from typing import List, Dict, Any, Optional

import numpy
import pandas
from matplotlib import pyplot

from ct_dirs import HNET_BASE_PATH
from data.vertebrae import VERTEBRAE
from hiwi import ImageList, Image
from lib.call_tool import call_tool
from lib.image_processing_tool import ImageProcessingTool
from lib.my_logger import logging
from lib.parameter_search import mean_confidence_interval_size
from lib.plot_modified_ey import load_and_plot
from lib.util import EBC
from load_data.load_image import image_by_patient_id
from load_data.update_iml_coordinates import UsingPredictedCoordinates
from model.hnet.eval_utils import classification_array_plot, per_vertebra_plot, patient_id_from_long_string, nii_metadata, closest_vertebra_within_range, world_coordinates, shift_distance, \
    dict_contains_vertebra, plot_landmarks


class HNetVersion(EBC):
    def __init__(self, exe_path: str, version_no: int, licensed_to_mac: Optional[str]):
        self.exe_path = exe_path
        self.version_no = version_no
        self.licensed_to_mac = licensed_to_mac


hnet_versions: List[HNetVersion] = [
    HNetVersion(HNET_BASE_PATH + '/2021-01-13-uksh-spine/dllTest/transfer_uksh_20210113.exe', 1, 'a85e455c74f0'),
    HNetVersion(HNET_BASE_PATH + "/2021-04-08-uksh-spine/dllTest/transfer_uksh_20210408.exe", 2, 'a85e455c74f0'),
    HNetVersion(HNET_BASE_PATH + "/2021-04-14-uksh-spine/dllTest/transfer_uksh_20210414.exe", 3, 'a85e455c74f0'),
    HNetVersion(HNET_BASE_PATH + "/2021-04-14-uksh-spine/dllTest/transfer_uksh_20210414_skipCoarse.exe", 4, 'a85e455c74f0'),
    HNetVersion(HNET_BASE_PATH + "/2021-10-05-uksh-spine/dllTest/transfer_uksh_20211005.exe", 5, 'a85e455c74f0'),  # currently best / Eren's Favourite
    # version 6 was skipped, would have been 2021-10-19-uksh-spine
    HNetVersion(HNET_BASE_PATH + "/2022-10-28-uksh-spine/dllTest/transfer_uksh_20221028.exe", 7, 'd8cb8a6fd5ec'),
    HNetVersion("./models/pbl/model_2020_01_23.pbl", 8, None),
]


def hnet_by_version(version_no: int):
    for version in hnet_versions:
        if version.version_no == version_no:
            return version


def flatten_dict(d) -> list:
    return [(*k, v) for k, v in d.items()]


def table_contains_vertebra(table, v):
    return (table['vertebra'] == v).any()


MPS_TEMPLATE = ''' 

    <?xml version="1.0" encoding="UTF-8" ?>
    <point_set_file>
        <file_version>0.1</file_version>
        <point_set>
            <time_series>
                <time_series_id>0</time_series_id>
                <Geometry3D ImageGeometry="false" FrameOfReferenceID="0">
                    <IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0"
                        m_0_2="0" m_1_0="0" m_1_1="1" m_1_2="0" m_2_0="0"
                        m_2_1="0" m_2_2="1" />
                    <Offset type="Vector3D" x="0" y="0" z="0" />
                    <Bounds>
                        <Min type="Vector3D" x="{0}" y="{1}"
                            z="{2}" />
                        <Max type="Vector3D" x="{3}" y="{4}"
                            z="{5}" />
                    </Bounds>
                </Geometry3D>
{6}
        
            </time_series>
        </point_set>
    </point_set_file>
'''


def to_mps(json_dict: dict):
    text = MPS_TEMPLATE
    positions = [v[0]['pos'] for v in json_dict.values()]
    if len(positions) > 0:
        min_coords = numpy.min(positions, axis=0)
        max_coords = numpy.max(positions, axis=0)
    else:
        min_coords = max_coords = (0, 0, 0)
    return text.format(*min_coords, *max_coords, ''.join(f'''
            <point>
                <id>{idx}</id>
                <specification>0</specification>
                <x>{pos[0]}</x>
                <y>{pos[1]}</y>
                <z>{pos[2]}</z>
            </point>''' for idx, pos in enumerate(positions)))


class EvaluationSettings(EBC):
    def __init__(self, exclude_empty_outputs: bool, excluded_vertebrae: List[int], distance_threshold_mm: float, exclude_center_6: bool, create_landmark_plots: bool):
        self.create_landmark_plots = create_landmark_plots
        self.exclude_center_6 = exclude_center_6
        self.exclude_empty_outputs = exclude_empty_outputs
        self.excluded_vertebrae = excluded_vertebrae
        self.distance_threshold_mm = distance_threshold_mm

    @classmethod
    def defaults(cls):
        return cls(
            exclude_empty_outputs=False,
            excluded_vertebrae=[],
            distance_threshold_mm=10.,
            exclude_center_6=False,
            create_landmark_plots=False,
        )


class HNet(ImageProcessingTool):
    def __init__(self, model_exe_path, model_version: int = None, results_dir=None, evaluation_settings=EvaluationSettings.defaults()):
        self.model_exe_path = model_exe_path
        self.evaluation_settings = evaluation_settings
        if results_dir is None:
            results_dir = os.path.join(tempfile.mkdtemp(), self.name())
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        if model_version is None:
            for v in hnet_versions:
                if os.path.basename(v.exe_path) == os.path.basename(model_exe_path):
                    model_version = v.version_no
                    break
            else:
                raise ValueError(
                    f"model_version {model_version} not found in `versions` variable. You can specify it there or in the constructor parameters explicitly with HNet(model_exe_path, model_version)")
        self.model_version = model_version

    def name(self) -> str:
        return os.path.basename(self.model_exe_path)

    def cleanup(self):
        shutil.rmtree(self.results_dir)

    def using_predicted_coordinates(self, iml):
        return UsingPredictedCoordinates(iml, self.name(), on_new_coordinates='add', on_missing_coordinates='invalidate')

    def evaluate(self, iml: ImageList):
        results_path = self.results_dir
        print('# Results path:', results_path)
        self.predict(iml)  # prediction is needed for the evaluation
        print('Loading labels...')
        labels = self.load_vertebra_centers(iml)
        all_vertebrae = []
        all_patients = []
        yilmaz_tp = mader_tp = {}
        yilmaz_fp = []
        yilmaz_fn = []
        yilmaz_unlabeled_tp = {}
        yilmaz_unlabeled_fp = []
        yilmaz_unlabeled_fn = []
        crashes = []
        mader_wrong_localizations_p_mis = {}
        glocker_distance_to_closest = {}
        glocker_identifications = []
        mader_all_correct = []
        yilmaz_all_correct_unlabeled = []
        mader_all_wrong = []
        yilmaz_all_wrong_unlabeled = []
        wrong_labels = []
        shifts: Dict[str, list] = {}
        ground_truth = []
        raw_results = []
        mader_fn = []
        mader_tn = []
        mader_fp = []
        ground_truth_counted_for_unlabeled_tp = {}
        coords: List[tuple] = []
        for filename in os.listdir(results_path):
            json_filepath = os.path.join(results_path, filename)
            if json_filepath.endswith('_cpr.json'):
                continue  # this is computed in a special modified view on the dataset
            if json_filepath.endswith('.json') and os.path.isfile(json_filepath):
                found_mader_mistake = False
                found_tp_or_tn = False
                found_unlabeled_fn_or_unlabeled_fp = False
                found_unlabeled_tp = False
                print(f'Checking {filename} ...')
                with open(json_filepath) as json_file:
                    predictions = json.load(json_file)
                try:
                    patient_id = int(patient_id_from_long_string(json_filepath))
                except IndexError:
                    print(f'WARNING: Could not determine patient id of {json_filepath}. Skipping this file.')
                    continue
                if len(predictions) == 0:
                    crashes.append(patient_id)
                    if self.evaluation_settings.exclude_empty_outputs:
                        continue
                img = image_by_patient_id(str(patient_id), iml)
                all_patients.append(patient_id)
                nii_filepath = str(img.path)
                relevant_labels = labels[labels['patient'] == str(patient_id)]
                if len(relevant_labels) == 0:
                    logging.warning(f'No labels found for patient {patient_id}')
                coordinate_transformer = nii_metadata(nii_filepath)
                # compare
                for vertebra, pos in predictions.items():
                    if vertebra in self.evaluation_settings.excluded_vertebrae:
                        continue

                    y_pred = pos[0]['pos']  # world coordinates
                    coords.append((patient_id, vertebra, *y_pred, os.path.basename(results_path)))
                    raw_results.append((patient_id, vertebra, *y_pred))
                    vertebra_coords = relevant_labels[relevant_labels['vertebra'] == vertebra]
                    close_vertebra, distance = closest_vertebra_within_range(ground_truth_vertebrae=relevant_labels,
                                                                             y_pred=y_pred,
                                                                             coordinate_transformer=coordinate_transformer)
                    _, distance_to_same_label = closest_vertebra_within_range(ground_truth_vertebrae=vertebra_coords,
                                                                              y_pred=y_pred,
                                                                              coordinate_transformer=coordinate_transformer)
                    glocker_distance_to_closest[patient_id, vertebra] = distance
                    assert vertebra_coords.shape[0] <= 1
                    if distance_to_same_label > self.evaluation_settings.distance_threshold_mm:
                        # predicted vertebra is not nearby
                        yilmaz_fp.append((patient_id, vertebra))
                    else:
                        # ground truth coordinate with same label nearby
                        mader_tp[patient_id, vertebra] = distance

                    if distance > self.evaluation_settings.distance_threshold_mm:
                        # no vertebra nearby
                        yilmaz_unlabeled_fp.append((patient_id, vertebra))
                        found_unlabeled_fn_or_unlabeled_fp = True
                    else:
                        if (patient_id, close_vertebra) not in ground_truth_counted_for_unlabeled_tp:
                            # ground truth coordinate with any label nearby
                            ground_truth_counted_for_unlabeled_tp[patient_id, close_vertebra] = distance
                            yilmaz_unlabeled_tp[patient_id, vertebra] = distance
                            found_unlabeled_tp = True
                        assert found_unlabeled_tp

                    if distance < self.evaluation_settings.distance_threshold_mm and close_vertebra != vertebra:
                        # there is a different vertebra nearby
                        wrong_labels.append((patient_id, vertebra, close_vertebra))

                    if distance < 20 and close_vertebra == vertebra:
                        # correct identification according to glocker et al
                        glocker_identifications.append((patient_id, vertebra))

                    if self.evaluation_settings.distance_threshold_mm < distance_to_same_label < math.inf:
                        # the vertebra is in the image, but somewhere else
                        mader_wrong_localizations_p_mis[patient_id, vertebra] = distance_to_same_label

                    if distance < self.evaluation_settings.distance_threshold_mm and vertebra in VERTEBRAE:
                        d = shift_distance(close_vertebra, vertebra)
                        if d > 0:
                            shifts.setdefault(f'shifted_by_{d:02d}', []).append((patient_id, vertebra, close_vertebra))
                        else:
                            assert (patient_id, vertebra) in mader_tp

                for _, row in relevant_labels.iterrows():
                    assert row['vertebra'] not in self.evaluation_settings.excluded_vertebrae
                    all_vertebrae.append((patient_id, row['vertebra']))
                    y_true = world_coordinates(coordinate_transformer, row)
                    ground_truth.append((patient_id, row['vertebra'], *y_true))
                    if (patient_id, row['vertebra']) not in mader_tp:
                        yilmaz_fn.append((patient_id, row['vertebra']))
                    if not any(math.sqrt((y_true[0] - y_pred[0]) ** 2 +
                                         (y_true[1] - y_pred[1]) ** 2 +
                                         (y_true[2] - y_pred[2]) ** 2) < self.evaluation_settings.distance_threshold_mm
                               for vertebra, pos in predictions.items()
                               if vertebra not in EXCLUDED_VERTEBRAE
                               for y_pred in [pos[0]['pos']]):
                        yilmaz_unlabeled_fn.append((patient_id, row['vertebra']))
                        found_unlabeled_fn_or_unlabeled_fp = True

                for vertebra in VERTEBRAE:
                    if vertebra in self.evaluation_settings.excluded_vertebrae:
                        continue
                    if table_contains_vertebra(relevant_labels, vertebra) and not dict_contains_vertebra(predictions, vertebra):
                        found_mader_mistake = True
                        mader_fn.append((patient_id, vertebra))
                    elif table_contains_vertebra(relevant_labels, vertebra) and dict_contains_vertebra(predictions, vertebra):
                        assert (patient_id, vertebra) in mader_tp or (patient_id, vertebra) in mader_wrong_localizations_p_mis
                        if (patient_id, vertebra) in mader_wrong_localizations_p_mis:
                            found_mader_mistake = True
                        else:
                            # (patient_id, vertebra) in mader_tp
                            found_tp_or_tn = True
                    elif not table_contains_vertebra(relevant_labels, vertebra) and dict_contains_vertebra(predictions, vertebra):
                        found_mader_mistake = True
                        mader_fp.append((patient_id, vertebra))
                    elif not table_contains_vertebra(relevant_labels, vertebra) and not dict_contains_vertebra(predictions, vertebra):
                        found_tp_or_tn = True
                        mader_tn.append((patient_id, vertebra))

                if not found_mader_mistake:
                    mader_all_correct.append(patient_id)
                if not found_unlabeled_fn_or_unlabeled_fp:
                    yilmaz_all_correct_unlabeled.append(patient_id)
                if not found_tp_or_tn:
                    mader_all_wrong.append(patient_id)
                if not found_unlabeled_tp:
                    yilmaz_all_wrong_unlabeled.append(patient_id)

                assert set(mader_tp).issubset(ground_truth_counted_for_unlabeled_tp)
                assert set(mader_tp).issubset(all_vertebrae)
                assert mader_tp is yilmaz_tp
                assert set(mader_wrong_localizations_p_mis).issubset(yilmaz_fp)
                assert set(yilmaz_unlabeled_fn).issubset(yilmaz_fn)
                assert set(yilmaz_unlabeled_fp).issubset(yilmaz_fp)
                assert set(yilmaz_fn).issubset(all_vertebrae)
                assert set(yilmaz_fp).isdisjoint(mader_tp), set(yilmaz_fp).intersection(mader_tp)
                assert set(mader_wrong_localizations_p_mis).isdisjoint(mader_tp)
                assert set(yilmaz_unlabeled_fp).isdisjoint(yilmaz_unlabeled_tp)
                assert len(ground_truth) >= len(yilmaz_unlabeled_fn) + len(yilmaz_unlabeled_tp)  # can be larger if a vertebra was excluded
                assert len(ground_truth) == len(yilmaz_fn) + len(yilmaz_tp)
                assert len(ground_truth) == len(mader_fn) + len(mader_tp) + len(mader_wrong_localizations_p_mis)
                assert len(mader_fn) + len(mader_tp) + len(mader_wrong_localizations_p_mis) + len(mader_fp) + len(mader_tn) \
                       == len(all_patients) * (len(VERTEBRAE) - len(self.evaluation_settings.excluded_vertebrae))

                # create some nice plot like AOM
                if self.evaluation_settings.create_landmark_plots:
                    lm_plot_paths = [json_filepath.replace('.json', '.png'),
                                     json_filepath.replace('.json', '.svg')]
                    if all(os.path.isfile(lm_plot_path) for lm_plot_path in lm_plot_paths):
                        continue
                    print('  -> plotting landmarks...')
                    pyplot.figure()
                    # noinspection PyTypeChecker
                    plot_landmarks(image_path=nii_filepath,
                                   patient_id=patient_id,
                                   y_true=ground_truth,
                                   y_pred=raw_results,
                                   tool_name=self.natural_name())
                    for lm_plot_path in lm_plot_paths:
                        pyplot.savefig(lm_plot_path, transparent=True)
                    # pyplot.show()
                    pyplot.close()

        os.makedirs(os.path.join(results_path, 'summary'), exist_ok=True)

        # dump coordinates as CSV
        header = ('studyDescription', 'vertebra', *(column for suffix in ['Center'] for column in [f'{suffix}X', f'{suffix}Y', f'{suffix}Z', 'source']))
        csv_filename = os.path.join(results_path, 'summary', 'real_world_coordinates.csv')
        with open(csv_filename, 'w') as f:
            coords.insert(0, header)
            f.write('\n'.join(';'.join(str(e) for e in c) for c in coords))
        print(f'Wrote {csv_filename}')

        # actually compute metrics
        ec = self.filter_excluded_centers
        yilmaz_recall = len(ec(mader_tp)) / (len(ec(yilmaz_tp)) + len(ec(yilmaz_fn)))
        mader_recall = len(ec(mader_tp)) / (len(ec(mader_tp)) + len(ec(mader_fn)))
        yilmaz_precision = len(ec(mader_tp)) / (len(ec(mader_tp)) + len(ec(yilmaz_fp)))
        if (len(ec(mader_tp)) + len(ec(mader_fp))) == 0:
            mader_precision = 0
        else:
            mader_precision = len(ec(mader_tp)) / (len(ec(mader_tp)) + len(ec(mader_fp)))
        yilmaz_unlabeled_recall = len(ec(yilmaz_unlabeled_tp)) / (len(ec(yilmaz_unlabeled_tp)) + len(ec(yilmaz_unlabeled_fn)))
        yilmaz_unlabeled_precision = len(ec(yilmaz_unlabeled_tp)) / (len(ec(yilmaz_unlabeled_tp)) + len(ec(yilmaz_unlabeled_fp)))
        localization_errors = list(ec(mader_tp).values())
        localization_error_tp = {'mean': numpy.mean(localization_errors),
                                 'std': numpy.std(localization_errors),
                                 '95%conf': mean_confidence_interval_size(localization_errors), }
        localization_errors += list(ec(mader_wrong_localizations_p_mis).values())
        glocker_localization_error = {'mean': numpy.mean(localization_errors),
                                      'std': numpy.std(localization_errors),
                                      '95%conf': mean_confidence_interval_size(localization_errors), }
        mader_success_rate = (len(ec(mader_tn)) + len(ec(mader_tp))) / (len(ec(all_patients)) * (len(VERTEBRAE) - len(self.evaluation_settings.excluded_vertebrae)))
        glocker_identification_rate = len(ec(glocker_identifications)) / len(ec(ground_truth))

        if yilmaz_precision == 0 or yilmaz_recall == 0:
            yilmaz_f1 = 0
        else:
            yilmaz_f1 = 2 / (1 / yilmaz_precision + 1 / yilmaz_recall)
        if mader_precision == 0 or mader_recall == 0:
            mader_f1 = 0
        else:
            mader_f1 = 2 / (1 / mader_precision + 1 / mader_recall)

        results = {
            'comments': {
                'all_vertebrae': 'All ground truth vertebrae visible in the image.',
                'all_patients': 'Patient ids for all images.',
                'crashes': 'Patient ids where the tool did not find any vertebrae',
                'yilmaz_tp': f'Same as mader_tp',
                'yilmaz_fn': f'Ground truth coordinates that do not have a predicted position with the same label within {self.evaluation_settings.distance_threshold_mm} mm',
                'yilmaz_fp': f'Predicted positions where the ground truth with the same label is not within {self.evaluation_settings.distance_threshold_mm} mm',
                'mader_fn': f'Landmark exists, but is not detected',
                'mader_tn': f'Landmark missing and not detected',
                'mader_fp': f'Landmark missing, but detected',
                'mader_tp': f'Predicted position where the ground truth coordinate with the same label is closer than {self.evaluation_settings.distance_threshold_mm} mm. '
                            f'The last entry is the distance in mm.',
                'yilmaz_unlabeled_tp': f'Predicted positions that have a ground truth coordinate with any label within {self.evaluation_settings.distance_threshold_mm} mm. '
                                       f'The last entries are the distance in mm and the correct label. Each ground truth coordinate is counted at most once.',
                'yilmaz_unlabeled_fn': f'Ground truth coordinates that do not have a predicted position within {self.evaluation_settings.distance_threshold_mm} mm',
                'yilmaz_unlabeled_fp': f'Predicted positions that do not have a ground truth coordinate within {self.evaluation_settings.distance_threshold_mm} mm',
                'glocker_identifications': f'closest centroid in the expert annotation corresponds to the correct vertebra and is within 20 mm',
                'glocker_distance_to_closest': f'distance to the closest ground truth coordinate without any label',
                'mader_wrong_localizations_p_mis': f'Predicted positions where the vertebra is in the image but not within {self.evaluation_settings.distance_threshold_mm} mm',
                'wrong_labels': f'Predicted coordinates where a ground truth coordinate is within {self.evaluation_settings.distance_threshold_mm} mm '
                                f'but not with the same label. The last entry is the correct label of the closest vertebra.',
                'shift_n': f'Vertebrae that were predicted to a location where actually another vertebra is that is. '
                           f'The number n represents the distance in levels, for example the distance between L2 and T12 is {shift_distance("L2", "T12")}. '
                           f'The last entry is the correct label of the closest vertebra.',
                'mader_all_correct': f'patients without mader_fn, mader_fp, mader_wrong_localizations_p_mis. '
                                     f'This is equivalent to listing patients without yilmaz_fn and yilmaz_fp.',
                'yilmaz_all_correct_unlabeled': f'patients without yilmaz_unlabeled_fp and yilmaz_unlabeled_fn',
                'mader_all_wrong': f'patients without mader_tp',
                'yilmaz_all_wrong_unlabeled': f'patients without yilmaz_unlabeled_tp',
                'raw_results': f'Predicted positions (output of the localization tool)',
                'ground_truth': f'Ground truth coordinates as annotated by TF with SpineAnalyzer (y and z-coordinates: longitudinal, anteroposterior) '
                                f'and by EBY (x-coordinate, lateral) with the help of an earlier version of PBL.',
                'yilmaz_sensitivity': 'yilmaz_tp / (yilmaz_tp + yilmaz_fn)',
                'yilmaz_precision': 'yilmaz_tp / (yilmaz_tp + yilmaz_fp)',
                'yilmaz_F1': '2 / (1 / yilmaz_precision + 1 / yilmaz_recall)',
                'yilmaz_unlabeled_sensitivity': 'yilmaz_unlabeled_tp / (yilmaz_unlabeled_tp + yilmaz_unlabeled_fn)',
                'yilmaz_unlabeled_precision': 'yilmaz_unlabeled_tp / (yilmaz_unlabeled_tp + yilmaz_unlabeled_fp)',
                'yilmaz_unlabeled_F1': '2 / (1 / yilmaz_unlabeled_precision + 1 / yilmaz_unlabeled_recall)',
                'mader_sensitivity': 'mader_tp / (mader_tp + mader_fn)',
                'mader_precision': 'mader_tp / (mader_tp + mader_fp)',
                'mader_F1': '2 / (1 / mader_precision + 1 / mader_recall)',
                'localization_error_tp': f'average distance of a correct (within {self.evaluation_settings.distance_threshold_mm} mm) predicted position to the ground_truth vertebra',
                'ground_truth_counted_for_unlabeled_tp': f'helper array that stores the ground truth coordinates with nearby predictions.',
                'glocker_localization_error': f'Distance of each prediction to the corresponding ground truth',
                'glocker_identification_rate': f'len(glocker_identifications) / len(ground_truth_vertebrae)',
                'mader_success_rate': f'(len(mader_tn) + len(mader_tp)) / (len(all_patients) * len(VERTEBRAE))',
            },
            'values': {
                'all_vertebrae': all_vertebrae,
                'all_patients': all_patients,
                'crashes': crashes,
                'yilmaz_tp': yilmaz_tp,
                'yilmaz_fn': yilmaz_fn,
                'yilmaz_fp': yilmaz_fp,
                'yilmaz_unlabeled_fn': yilmaz_unlabeled_fn,
                'yilmaz_unlabeled_fp': yilmaz_unlabeled_fp,
                'yilmaz_unlabeled_tp': flatten_dict(yilmaz_unlabeled_tp),
                'mader_fn': mader_fn,
                'mader_tn': mader_tn,
                'mader_fp': mader_fp,
                'mader_tp': flatten_dict(mader_tp),
                'mader_wrong_localizations_p_mis': flatten_dict(mader_wrong_localizations_p_mis),
                'glocker_identifications': glocker_identifications,
                'glocker_distance_to_closest': flatten_dict(glocker_distance_to_closest),
                'wrong_labels': wrong_labels,
                **{k: v for k, v in sorted(shifts.items())},
                'mader_all_correct': mader_all_correct,
                'yilmaz_all_correct_unlabeled': yilmaz_all_correct_unlabeled,
                'mader_all_wrong': mader_all_wrong,
                'yilmaz_all_wrong_unlabeled': yilmaz_all_wrong_unlabeled,
                'raw_results': raw_results,
                'ground_truth': ground_truth,
                'ground_truth_counted_for_unlabeled_tp': ground_truth_counted_for_unlabeled_tp,
            },
            'aggregates': {
                'yilmaz_sensitivity': yilmaz_recall,
                'yilmaz_precision': yilmaz_precision,
                'yilmaz_F1': yilmaz_f1,
                'yilmaz_unlabeled_sensitivity': yilmaz_unlabeled_recall,
                'yilmaz_unlabeled_precision': yilmaz_unlabeled_precision,
                'yilmaz_unlabeled_F1': 2 / (1 / yilmaz_unlabeled_precision + 1 / yilmaz_unlabeled_recall),
                'mader_sensitivity': mader_recall,
                'mader_precision': mader_precision,
                'mader_F1': mader_f1,
                '*glocker_localization_error*': glocker_localization_error,
                'localization_error_tp': localization_error_tp,
                '*glocker_identification_rate*': glocker_identification_rate,
                '*mader_success_rate*': mader_success_rate,
            },
        }
        for v in results['values'].values():
            assert len(set(v)) == len(v), set([(x, v.count(x)) for x in v if v.count(x) > 1])

        def count(xs):
            return len(xs)

        def distinct_patient_count(xs):
            return len(set(x if isinstance(x, int) else x[0]
                           for x in xs))

        for aggregate in [count, distinct_patient_count, sorted, ]:
            results[aggregate.__name__] = {
                k: aggregate(v)
                for k, v in list(results['values'].items())
            }
        del results['values']  # redundant: is already in the 'sorted' key

        # store in summary file
        with open(os.path.join(results_path,
                               'summary',
                               'DiagBilanz_evaluation_results.json'), 'w') as f:
            json.dump(results, f)

        # pyplot.show()
        pyplot.close()

        per_vertebra_plot(results=results, exclude_center_6=self.evaluation_settings.exclude_center_6)
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'error_distribution_vertebrae.png'), transparent=True)
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'error_distribution_vertebrae.svg'), transparent=True)
        # pyplot.show()
        pyplot.close()

        classification_array_plot(results=results, simple=False, format_for_mlmi=True,
                                  exclude_empty_outputs=self.evaluation_settings.exclude_empty_outputs,
                                  excluded_vertebrae=self.evaluation_settings.excluded_vertebrae)
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'mlmi_classification_array_plot.png'), transparent=True)
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'mlmi_classification_array_plot.svg'), transparent=True)
        # pyplot.show()
        pyplot.close()

        #  plot results
        classification_array_plot(results=results, simple=False,
                                  exclude_empty_outputs=self.evaluation_settings.exclude_empty_outputs,
                                  excluded_vertebrae=self.evaluation_settings.excluded_vertebrae)
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'classification_array_plot.svg'), transparent=True)
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'classification_array_plot.png'), transparent=True)
        # pyplot.show()
        pyplot.close()

        classification_array_plot(results=results, simple=True,
                                  exclude_empty_outputs=self.evaluation_settings.exclude_empty_outputs,
                                  excluded_vertebrae=self.evaluation_settings.excluded_vertebrae)
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'simple_classification_array_plot.svg'), transparent=True)
        pyplot.savefig(os.path.join(results_path,
                                    'summary',
                                    'simple_classification_array_plot.png'), transparent=True)
        # pyplot.show()
        pyplot.close()

    def filter_excluded_centers(self, data):
        if not self.evaluation_settings.exclude_center_6:
            return data
        if isinstance(data, dict):
            return {
                k: v
                for k, v in data.items()
                if not self.belongs_to_center_6(k)
            }
        if isinstance(data, list):
            return [
                x
                for x in data
                if not self.belongs_to_center_6(x)
            ]
        raise ValueError

    @staticmethod
    def belongs_to_center_6(x):
        if isinstance(x, int):
            return str(x).startswith('6')
        if isinstance(x, str):
            return x.startswith('6')
        if isinstance(x, tuple):
            assert isinstance(x[0], int)
            return str(x[0]).startswith('6')
        raise ValueError

    def predict_on_single_image(self, img: Image) -> dict:
        input_file = str(img.path)
        input_file = self.infer_nii_path_if_necessary(input_file)

        nii_filename = os.path.basename(input_file)

        result_name = self.result_name_for_input(nii_filename)
        tool = self.model_exe_path
        command = [os.path.abspath(tool), os.path.abspath(input_file), os.path.abspath(result_name)]
        if 'skipCoarse' in tool:
            command.append('-skipCoarse')

        ext = '.nii.gz'
        assert input_file.endswith(ext)
        json_result_path = os.path.join(os.path.abspath(result_name) + '_spine_fine.json')

        try:
            if os.path.isfile(json_result_path):
                print(json_result_path, 'already exists. Using the existing file.')
            else:
                call_tool(command, force_cpu=True, cwd=os.path.abspath(os.path.dirname(os.path.dirname(tool))))
        except CalledProcessError as e:
            print(e)
            print('Ignoring a failed process call and dumping empty json file..')
            with open(json_result_path, 'w') as f:
                json.dump({}, f)
        assert os.path.isfile(json_result_path), json_result_path
        mps_path = self.json_file_to_mps_file(json_result_path)

        outputs = self.predictions_from_json_file(json_result_path)

        cpr_result_path: Any = json_result_path.replace('_spine_fine.json', '_spine_cpr.nii.gz')
        if os.path.isfile(cpr_result_path):
            load_and_plot(cpr_result_path)
            outputs['_cpr_result_path'] = cpr_result_path

        cpr_overlay_path: Any = json_result_path.replace('_spine_fine.json', '_spine_cpr_overlay.png')
        if os.path.isfile(cpr_overlay_path):
            outputs['_cpr_overlay_path'] = cpr_overlay_path

        cpr_coordinates_path: Any = json_result_path.replace('_spine_fine.json', '_spine_cpr.json')
        if os.path.isfile(cpr_coordinates_path):
            outputs['_cpr_coordinates_path'] = cpr_coordinates_path

        assert os.path.isfile(mps_path)
        outputs['_mps_path'] = mps_path

        return outputs

    def result_name_for_input(self, nii_filename):
        result_name = os.path.join(self.results_dir, os.path.basename(nii_filename))
        assert result_name.endswith('.nii.gz')
        result_name = result_name[:-len('.nii.gz')]
        return result_name

    def json_file_to_mps_file(self, json_result_path):
        predictions = self.predictions_from_json_file(json_result_path)
        mps_text = to_mps(predictions)
        mps_result_filename = json_result_path.replace('.json', '.mps')
        with open(mps_result_filename, 'w') as mps_file:
            mps_file.write(mps_text)
        assert os.path.isfile(mps_result_filename)
        return mps_result_filename

    @staticmethod
    def predictions_from_json_file(json_path) -> Dict[str, List[Dict[str, List[float]]]]:
        with open(json_path) as json_file:
            predictions = json.load(json_file)
        return predictions

    def natural_name(self):
        return f'hNet (Version {self.model_version})'

    def load_vertebra_centers(self, iml) -> pandas.DataFrame:
        coordinates = [
            {
                'patient': img['patient_id'],
                'vertebra': vertebra,
                'CenterX': img.parts[vertebra].position[0],
                'CenterY': img.parts[vertebra].position[1],
                'CenterZ': img.parts[vertebra].position[2],
            }
            for img in iml
            for vertebra in img.parts
        ]
        return pandas.DataFrame.from_records(coordinates)
