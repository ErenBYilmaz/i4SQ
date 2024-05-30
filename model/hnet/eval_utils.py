import math
import re
from typing import Union, Tuple, List

import SimpleITK
import matplotlib.patches
import numpy
from matplotlib import pyplot
from matplotlib.axes import Axes
import matplotlib.patheffects as PathEffects

from data.vertebrae import VERTEBRAE


def world_coordinates(coordinate_transformer, coords_row):
    true_voxel_coordinates = numpy.array((
        coords_row['CenterX'],
        coords_row['CenterY'],
        coords_row['CenterZ'],
    ))
    if true_voxel_coordinates.shape == (3, 1):
        true_voxel_coordinates = numpy.squeeze(true_voxel_coordinates, axis=1)
    assert true_voxel_coordinates.shape == (3,)
    transform = coordinate_transformer.TransformContinuousIndexToPhysicalPoint
    y_true = transform(true_voxel_coordinates)  # true world coordinates
    return y_true


def closest_vertebra_within_range(ground_truth_vertebrae,
                                  y_pred,
                                  coordinate_transformer):
    min_d = math.inf
    result = None
    for v2 in ground_truth_vertebrae.iterrows():
        y_true = world_coordinates(coordinate_transformer, v2[1])
        distance = math.sqrt((y_true[0] - y_pred[0]) ** 2 +
                             (y_true[1] - y_pred[1]) ** 2 +
                             (y_true[2] - y_pred[2]) ** 2)
        if distance < min_d:
            min_d = distance
            result = v2[1]['vertebra']

    return result, min_d


def empty_array_plot(axis, results, format_for_mlmi=False, vertebrae=None, patients=None, exclude_center_6=None):
    """
    :param results: can be None if patients is not None
    """
    if exclude_center_6 is None:
        exclude_center_6 = format_for_mlmi
    if vertebrae is None:
        vertebrae = VERTEBRAE
    if patients is None:
        if results is None:
            raise ValueError
        patients = sorted(set(results['sorted']['all_patients'] + results['sorted']['crashes']))
    if exclude_center_6:
        patients = [p for p in patients if not str(p).startswith('6')]
    if axis is None:
        fig = pyplot.figure(figsize=(15, 4), dpi=202)
        ax = fig.add_subplot(111)
    else:
        ax = axis
    ax: Axes
    # lines between centers
    lines = []
    for p_idx, p in enumerate(patients):
        if str(p).endswith('01'):  # TODO test
            lines.append(p_idx - 0.5)
    ax.vlines(lines, colors='black', ymin=-0.5, ymax=len(vertebrae) - 0.5)
    ax.set_xlim(-0.5, len(patients) - 0.5)

    if not format_for_mlmi:
        ax.set_xlabel(f'{len(patients)} cases')
    x_ticks = range(0, len(patients), 6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([patients[x] for x in x_ticks])
    ax.set_ylim(-0.5, len(vertebrae) - 0.5)
    if not format_for_mlmi:
        ax.set_ylabel(f'{len(vertebrae)} vertebrae')
    ax.invert_yaxis()
    y_ticks = range(0, len(vertebrae), 4)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([vertebrae[y] for y in y_ticks])
    ax.set_aspect('equal', adjustable='box')
    return ax, patients


def outputs_plot(results, outputs, format_for_mlmi=False, axis=None):
    if format_for_mlmi and axis is not None:
        raise ValueError
    print('Plotting pipeline putputs...')
    if format_for_mlmi:
        ordered_vertebrae = (
                ['T' + str(i) for i in range(4, 13)]
                + ['L' + str(i) for i in range(1, 5)]
        )
    else:
        ordered_vertebrae = VERTEBRAE
    patients = sorted(set(results['sorted']['all_patients'] + results['sorted']['crashes']))
    if format_for_mlmi:
        patients = [p for p in patients if not str(p).startswith('6')]
    if format_for_mlmi:
        fig, axes = pyplot.subplots(2, figsize=(7.5, 3), dpi=202)
        plots = [(patients[s], ax) for s, ax in zip([slice(len(patients) // 2 + 1), slice(len(patients) // 2 + 1, len(patients))], axes)]
        for ps, ax in plots:
            empty_array_plot(ax, results, format_for_mlmi=format_for_mlmi, vertebrae=ordered_vertebrae, patients=ps)
    else:
        ax, patients = empty_array_plot(axis, results, format_for_mlmi=format_for_mlmi, vertebrae=ordered_vertebrae, patients=patients)
        plots = [(patients, ax)]
    to_plot_coordinate = lambda x: x * 23
    from_plot_coordinate = lambda x: x / 23
    # y = []

    for ps, ax in plots:
        fracture_xs = []
        fracture_ys = []
        nonfracture_xs = []
        nonfracture_ys = []

        for p_idx, p in enumerate(ps):
            if p not in outputs:
                continue
            # y.append(outputs[p]['y_pred'])
            if outputs[p]['y_true']:
                fracture_xs.append(p_idx)
                fracture_ys.append(outputs[p]['y_pred'])
            else:
                nonfracture_xs.append(p_idx)
                nonfracture_ys.append(outputs[p]['y_pred'])

        # y = numpy.array(y)
        fracture_xs = numpy.array(fracture_xs)
        fracture_ys = numpy.array(fracture_ys)
        nonfracture_xs = numpy.array(nonfracture_xs)
        nonfracture_ys = numpy.array(nonfracture_ys)

        # pyplot.plot(to_plot_coordinate(y))
        ax.scatter(fracture_xs, to_plot_coordinate(fracture_ys), facecolors='black', edgecolors='grey', label='At least one OF')
        ax.scatter(nonfracture_xs, to_plot_coordinate(nonfracture_ys), facecolors='none', edgecolors='grey', label='No OF')
        if ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_yticks([to_plot_coordinate(y) for y in [0, 0.25, 0.5, 0.75, 1]])
        ax.set_yticklabels([from_plot_coordinate(y) for y in ax.get_yticks()])
        ax.set_ylabel('Pipeline output')
        ax.legend()


def color_variant(target_hex, offset=50):
    if len(target_hex) != 7:
        return None
    hex_color = [target_hex[x:x + 2] for x in [1, 3, 5]]
    new_rgb = [int(hex_value, 16) + offset for hex_value in hex_color]
    new_rgb = [min([255, max([0, i])]) for i in new_rgb]

    return '#%02x%02x%02x' % (new_rgb[0], new_rgb[1], new_rgb[2])


def per_vertebra_plot(results, exclude_center_6: bool):
    """
    modified from AOM's results_cases.py/error_distribution
    """
    print('Plotting Results per vertebra level...')
    VALUES = [
        'mader_tp',
        'mader_wrong_localizations_p_mis',
        'mader_fn',
        'mader_fp',
        'mader_tn',
    ]
    distribution = numpy.zeros((len(VERTEBRAE), len(VALUES)))
    LANDMARKS = BAR_NAMES = VERTEBRAE
    for value in VALUES:
        for row in results['sorted'][value]:
            p, v = row[0], row[1]
            if str(p).startswith('6') and exclude_center_6:
                continue
            distribution[LANDMARKS.index(v), VALUES.index(value)] += 1
    distribution /= distribution.sum(axis=1)[:, numpy.newaxis]
    distribution = numpy.nan_to_num(distribution, nan=0)
    distribution *= 100
    VALUE_NAMES = [
        'LM exists, detected and localized (TP)',
        'LM exists, detected but mis-localized (Pmis)',
        'LM exists, but not detected (FN)',
        'LM missing, but detected (FP)',
        'LM missing, and not detected (TN)',
    ]
    # if verbos -localization rate: {distribution[i, 4]}')
    pyplot.rc('xtick', **{'major.pad': 4})
    pyplot.figure(figsize=(8, 4), dpi=300)
    pyplot.subplots_adjust(top=0.97, bottom=0.18, right=0.99, left=0.12,
                           wspace=0, hspace=0)
    pyplot.gca().xaxis.grid(False)
    pyplot.gca().spines['right'].set_visible(False)
    pyplot.gca().spines['top'].set_visible(False)
    BAR_WIDTH = 1.1
    BAR_DIST = 2
    BAR_POSITIONS = numpy.arange(len(BAR_NAMES)) * BAR_DIST
    # [0, 1.5, 3, 4.5, 7.5, 9, 10.5, 12]
    all_first_colors = [next(pyplot.gca()._get_lines.prop_cycler) for i in range(5)]
    nth_color = lambda n: all_first_colors[n]['color']
    colors = [
        color_variant(nth_color(0), 50),
        color_variant(nth_color(3), 100),
        nth_color(3),
        color_variant(nth_color(3), -100),
        color_variant(nth_color(0), -50),
    ]
    # hatches = ['...', '---', None, '///', '|||']
    # colors = ['#d86f6f', '#a42e2e', '#ef1c1c',
    #              '#5ba3d1', '#0d79be']

    # Actual data
    bars = [pyplot.bar(BAR_POSITIONS, distribution[:, i], BAR_WIDTH,
                       bottom=distribution[:, :i].sum(axis=1), color=colors[i],
                       # edgecolor=[color_variant(colors[i], 50)] * distribution.shape[0],
                       # linewidth=0,
                       # hatch=hatches[i]
                       )[0]
            for i in range(len(VALUES))]

    # ground truth division
    pyplot.step(BAR_POSITIONS, distribution[:, :3].sum(axis=1), color='black', where='mid')
    pyplot.legend(bars[::-1], VALUE_NAMES[::-1], loc='lower left',
                  fontsize='x-small', handlelength=1.5)
    pyplot.ylabel('Landmarks · test images (%)')
    pyplot.ylim(0, 100)
    pyplot.xticks(BAR_POSITIONS, BAR_NAMES, fontsize='x-small')
    for label in pyplot.gca().get_xticklabels():
        # print('label', label)
        if label.get_text() == 'All':
            # print('Bold')
            label.set_name('DejaVu Serif')
            label.set_weight('semibold')
            # print('d', label.get_fontproperties())
            # print('dd', label.get_fontproperties().get_name())
    # pyplot.savefig('error_distribution.pdf', transparent=True, frameon=False)
    # pyplot.close()


def is_fractured(patient, vertebra, df):
    if df is None:
        return None
    df = df[df['Patients Name'] == int(patient)]
    df = df[df['Label'] == vertebra]
    if df.empty:
        return False
    elif len(df) == 1:
        return df['Differential Diagnosis Category'].item() == 'Osteoporotic Fracture'
    else:
        raise AssertionError


def is_deformity(patient, vertebra, df):
    if df is None:
        return None
    df = df[df['Patients Name'] == int(patient)]
    df = df[df['Label'] == vertebra]
    if df.empty:
        return False
    elif len(df) == 1:
        return df['Differential Diagnosis Category'].item() == 'Deformity'
    else:
        raise AssertionError


def shift_distance(close_vertebra, vertebra):
    return abs(VERTEBRAE.index(close_vertebra) - VERTEBRAE.index(vertebra))


def table_contains_vertebra(table, v):
    return (table['vertebra'] == v).any()


def dict_contains_vertebra(json_dict, v):
    return v in json_dict


PatientId = int
Vertebra = str
X = Y = Z = float


def plot_landmarks(image_path: str,
                   patient_id: PatientId,
                   tool_name: str,
                   y_true: List[Tuple[PatientId, Vertebra, X, Y, Z]],
                   y_pred: List[Tuple[PatientId, Vertebra, X, Y, Z]]):
    import hiwi.plot
    img: SimpleITK.Image = SimpleITK.ReadImage(image_path)
    landmarks = {v: {} for p, v, x, y, z in y_true + y_pred
                 if p == patient_id}
    for p, v, x, y, z in y_true:
        if p != patient_id:
            continue
        landmarks[v][hiwi.plot.Landmark.TRUE_POS] = numpy.array(img.TransformPhysicalPointToContinuousIndex((x, y, z)))
    for p, v, x, y, z in y_pred:
        if p != patient_id:
            continue
        landmarks[v][hiwi.plot.Landmark.PRED_POS] = numpy.array(img.TransformPhysicalPointToContinuousIndex((x, y, z)))
    hiwi.plot.landmarks(image=SimpleITK.GetArrayFromImage(img),
                        landmarks=landmarks,
                        title=f'{tool_name} on {patient_id}',
                        views=['yz'],
                        invert_axes='z',
                        spacing=numpy.array(img.GetSpacing()))


def patient_id_from_long_string(ct_path) -> str:
    try:
        return re.findall(r'({0}\d\d\d)_'.format(site_id_from_long_string(ct_path)[-1]), ct_path)[0]
    except IndexError:
        try:
            return re.findall(r'({0}\d\d\d)'.format(site_id_from_long_string(ct_path)[-1]), ct_path)[0]
        except IndexError:
            return re.findall(r'(?:\b|_)(\d{10})(?:\b|_)', ct_path)[-1]  # trauma rad-number format


def site_id_from_long_string(ct_path) -> str:
    try:
        return re.findall(r'(?:^|[\\/_])([1345689])\d\d\d_', ct_path)[-1]
    except IndexError:
        return re.findall(r'(?:^|[\\/_])([1345689])\d\d\d', ct_path)[-1]


def classification_array_plot(results, simple=True, axis=None, format_for_mlmi=False, exclude_empty_outputs: bool = False, plot_fracture_backgrounds: bool = False, excluded_vertebrae: list = None):
    if excluded_vertebrae is None:
        excluded_vertebrae = []
    if format_for_mlmi and simple:
        raise ValueError
    if format_for_mlmi and axis is not None:
        raise ValueError
    print('Plotting Squares...')
    if format_for_mlmi:
        ordered_vertebrae = (
                ['T' + str(i) for i in range(4, 13)]
                + ['L' + str(i) for i in range(1, 5)]
        )
    else:
        ordered_vertebrae = VERTEBRAE
    patients = sorted(set(results['sorted']['all_patients'] + results['sorted']['crashes']))
    if format_for_mlmi:
        patients = [p for p in patients if not str(p).startswith('6')]
    # if format_for_mlmi:
    #     fig, axes = pyplot.subplots(2, figsize=(7.5, 3), dpi=202)
    #     plots = [(patients[s], ax) for s, ax in zip([slice(len(patients) // 2 + 1), slice(len(patients) // 2 + 1, len(patients))], axes)]
    #     for ps, ax in plots:
    #         empty_array_plot(ax, results, format_for_mlmi=format_for_mlmi, vertebrae=ordered_vertebrae, patients=ps)
    # else:
    ax, patients = empty_array_plot(axis, results, format_for_mlmi=format_for_mlmi, vertebrae=ordered_vertebrae, patients=patients)
    plots = [(patients, ax)]

    w1 = 10 / 18
    w2 = 14 / 18
    stroke_lw = 0.75
    text_padding = (0.125, -0.019)

    shifts = {}
    for k in results['sorted']:
        if k.startswith('shifted_by_'):
            n = re.fullmatch(r'shifted_by_0*(\d+)', k).group(1)
            if len(n) >= 2:  # two digits
                n = 'X'
            for patient_id, vertebra, close_vertebra in results['sorted'][k]:
                shifts[patient_id, vertebra] = n

    for ps, ax in plots:
        first_two_columns = lambda xs: list((x[0], x[1]) for x in xs)
        if exclude_empty_outputs:
            assert set(results['sorted']['all_patients']).isdisjoint(results['sorted']['crashes'])
        if plot_fracture_backgrounds:
            data = excel_panda(FRACTURE_LABELS_PATH, header=1)
        else:
            data = None

        for patient_idx, p in enumerate(ps):
            for vertebra_idx, v in enumerate(ordered_vertebrae):
                assert vertebra_idx == ordered_vertebrae.index(v)
                assert patient_idx == ps.index(p), (patient_idx, ps.index(p))
                color = None
                if not simple:
                    bg_color = None
                    if is_fractured(p, v, data):
                        assert bg_color is None
                        bg_color = 'black'
                    elif is_deformity(p, v, data) and not format_for_mlmi:
                        assert bg_color is None
                        bg_color = 'grey'
                    else:
                        assert bg_color is None
                        bg_color = 'white'
                    if bg_color != 'white':
                        rect = matplotlib.patches.Rectangle((patient_idx - w2 / 2, vertebra_idx - w2 / 2), w2, w2, color=bg_color)
                        ax.add_patch(rect)
                if v in excluded_vertebrae or (p in results['sorted']['crashes'] and exclude_empty_outputs):
                    assert color is None
                    color = 'lightgrey'
                if (p, v) in results['sorted']['mader_tn']:
                    assert color is None
                    color = 'white'
                elif (p, v) in first_two_columns(results['sorted']['mader_tp']):
                    assert color is None
                    color = 'palegreen'
                elif (p, v) in results['sorted']['mader_fp']:
                    assert color is None
                    color = 'blue'
                elif (p, v) in first_two_columns(results['sorted']['mader_wrong_localizations_p_mis']):
                    assert color is None
                    color = 'orange'
                elif (p, v) in results['sorted']['mader_fn']:
                    assert color is None
                    color = 'red'
                assert color is not None
                if color != 'white':
                    rect = matplotlib.patches.Rectangle((patient_idx - w1 / 2, vertebra_idx - w1 / 2), w1, w1, color=color)
                    ax.add_patch(rect)

        if not simple and not format_for_mlmi:
            for (patient_id, vertebra), n in shifts.items():
                if patient_id not in ps:
                    # probably excluded at some point
                    continue
                vertebra_idx = ordered_vertebrae.index(vertebra)
                patient_idx = ps.index(patient_id)
                txt = ax.text(x=patient_idx - w1 / 2 + text_padding[0],
                              y=vertebra_idx + w1 / 2 + text_padding[1],
                              s=n,
                              fontsize=4.25)
                txt.set_path_effects([PathEffects.withStroke(linewidth=stroke_lw, foreground='w')])

    if simple:
        ax.set_title('green: TP - red: FN - blue: FP - white: TN - orange: mislocalization - light grey: not countede')
    elif format_for_mlmi:
        ax.set_title('green: TP - red: FN - blue: FP - white: TN - orange: mislocalization - black border: osteoporotic fracture')
    else:
        ax.set_title('green: TP - red: FN - blue: FP - white: TN - orange: mislocalization - light grey: not counted\n'
                     'black border: osteoporotic fracture - grey border: deformity - digit: "shift" - X: shift >= 10')


reader = SimpleITK.SimpleITK.ImageFileReader()
reader.LoadPrivateTagsOn()


def nii_metadata(nii_filepath, as_reader=False, two_d=False, return_size=False) \
        -> Union[Union[SimpleITK.Image, SimpleITK.ImageFileReader], Tuple[Union[SimpleITK.Image, SimpleITK.ImageFileReader], float]]:
    reader.SetFileName(nii_filepath)
    reader.ReadImageInformation()
    if as_reader:
        if return_size:
            return reader, reader.GetSize()
        else:
            return reader
    if two_d:
        dummy_image = SimpleITK.GetImageFromArray(numpy.empty((1, 1)))
    else:
        dummy_image = SimpleITK.GetImageFromArray(numpy.empty((1, 1, 1)))
    dummy_image.SetDirection(reader.GetDirection())
    dummy_image.SetSpacing(reader.GetSpacing())
    dummy_image.SetOrigin(reader.GetOrigin())
    if return_size:
        return dummy_image, reader.GetSize()
    else:
        return dummy_image
