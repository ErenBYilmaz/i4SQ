import inspect
import json
import os
from typing import List, Dict, Tuple, Union

import PIL.Image
import SimpleITK
import dill
import numpy
import pandas
import pydicom_seg
from matplotlib import pyplot
from matplotlib.transforms import IdentityTransform
from pdf2dcm import Pdf2EncapsDCM

from hiwi import ImageList, Image
from lib.image_processing_tool import ImageProcessingPipeline
from lib.main_wrapper import main_wrapper
from lib.my_logger import logging
from load_data import VERTEBRAE
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.fnet import FNet
from model.fnet.model_analysis.analyze_trained_models import split_name
from model.hnet.hnet import HNet, hnet_by_version
from tasks import VertebraTasks, BinaryClassificationTask


class HNetFNetPipeline(ImageProcessingPipeline):
    def __init__(self, hnet: HNet, fnet: FNet, results_subdir: str = ''):
        super(HNetFNetPipeline, self).__init__(tools=[hnet, fnet])
        self.results_subdir = results_subdir
        self.hnet = hnet
        self.fnet = fnet
        self.fnet.use_coordinates = self.hnet.using_predicted_coordinates

    def name(self) -> str:
        return 'HNetFNetPipeline'

    def evaluate_whole_pipeline(self, iml_with_intermediate_predictions_and_evaluations: ImageList):
        from model.hnet_fnet.fracture_evaluation import classify_fractures, coords_from_iml
        if self.whole_pipeline_evaluation_task_name is None:
            logging.info('Cannot evaluate whole pipeline, no task name given')
            return
        model, config = self.fnet.load_trained_model_for_evaluation()
        tasks = VertebraTasks.deserialize(config['tasks'])
        with self.hnet.using_predicted_coordinates(iml_with_intermediate_predictions_and_evaluations):
            classify_fractures(
                fracture_model_name=self.fnet.name(),
                test_list=iml_with_intermediate_predictions_and_evaluations,
                coordinates=coords_from_iml(iml_with_intermediate_predictions_and_evaluations),
                verbose=1,
                task=tasks.by_name('01v23_012v3_0v123'),
                results_path=self.results_path(),
                threshold=0.5,
                exclude_center_6=True,
            )

    def results_path(self):
        return os.path.join(self.hnet.results_dir, self.results_subdir, self.fnet.name())

    def predict_on_single_image(self, img: Image) -> dict:
        result = super(HNetFNetPipeline, self).predict_on_single_image(img)
        self.plot_result(img, result)
        return result

    def plot_result(self, img: Image, result: dict):
        m = MaskCreator(self, img, result)
        dirname, basename = os.path.split(self.hnet.results_dir)
        basename = f'6_{self.fnet.name()}'
        dirname = os.path.join(dirname, basename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        to_nii_path = os.path.join(dirname, 'mask.nii.gz')
        m.create_mask(to_nii_path)

    @classmethod
    def pipeline_for_demonstrator(cls, out_dir_path: str, base_model_dir='models'):
        model_dir = os.path.join(base_model_dir, 'trauma_early_stopping_by_total_loss/trauma_early_stopping_by_total_loss_loss')
        return cls(
            hnet=(HNet(model_exe_path=hnet_by_version(7).exe_path,
                       results_dir=os.path.join(out_dir_path, '5_hnet_coordinates'))),
            fnet=(FNet(model_path=os.path.join(model_dir, 'trauma_early_stopping_by_total_loss_2023-04-17 10_35_04.172937_fold2.h5'),
                       use_coordinates='should_be_set-by_pipeline_constructor')),
            results_subdir='',
        )


class MaskCreator:
    def __init__(self, pipeline: HNetFNetPipeline, img: Image, result: dict):
        self.pipeline = pipeline
        self.img = img
        self.result = result

    def base_dcm_path(self):
        return self.img['base_dcm_path']

    def create_mask(self, to_nii_path: str):
        assert to_nii_path.endswith('.nii.gz')
        tmp_mask_path = to_nii_path.replace('.nii.gz', '.png')
        from_nii_path = self.infer_nii_path_if_necessary()
        s_img: SimpleITK.Image = SimpleITK.ReadImage(from_nii_path)
        self.plot_using_pyplot(s_img, tmp_mask_path)
        tmp_pdf_path = tmp_mask_path.replace('.png', '.pdf')
        self.png_to_pdf(tmp_mask_path, tmp_pdf_path)
        dcm_present = len(self.dcm_paths(self.base_dcm_path())) > 0
        if dcm_present:
            self.pdf_to_dcm(tmp_pdf_path, self.some_dcm_slice(self.base_dcm_path()), suffix='.pdf.dcm')
        mask = self.binarize_pyplot_to_mask(tmp_mask_path)
        mask = self.resize_and_mark_corner(mask, s_img)
        mask_img = self.mask_to_image(mask, s_img)
        self.save_as_nii(mask_img, to_nii_path)
        if dcm_present:
            self.nii_to_dcm_seg(to_nii_path, self.base_dcm_path(), to_dcm_seg_path=to_nii_path.replace('.nii.gz', '.dcm'))

    def png_to_pdf(self, png_path, pdf_path):
        img = PIL.Image.open(png_path)
        img = img.convert('RGB')
        img.save(pdf_path, 'PDF', resolution=100.0)

    def pdf_to_dcm(self, pdf_path, template_dcm_path, suffix=".dcm"):
        converter = Pdf2EncapsDCM()
        converted_dcm = converter.run(path_pdf=pdf_path, path_template_dcm=template_dcm_path, suffix=suffix)
        print(converted_dcm)

    def save_as_nii(self, mask_img, to_nii_path):
        os.makedirs(os.path.dirname(to_nii_path), exist_ok=True)
        SimpleITK.WriteImage(mask_img, to_nii_path)

    def mask_to_image(self, mask, s_img):
        mask_img = SimpleITK.GetImageFromArray(mask)
        mask_img.SetSpacing(s_img.GetSpacing())
        mask_img.SetOrigin(s_img.GetOrigin())
        mask_img.SetDirection(s_img.GetDirection())
        return mask_img

    def resize_and_mark_corner(self, mask, s_img):
        # resize mask to match the original image
        a = SimpleITK.GetArrayFromImage(s_img)
        shape_z, shape_y, shape_x = a.shape
        mask = numpy.array(PIL.Image.fromarray(mask).resize(
            (shape_y, shape_z), resample=PIL.Image.BILINEAR
        ))
        mask = mask[..., numpy.newaxis]  # add x axis
        # mark lower frontal corner, so that we can see if the mask was loaded correctly
        mask[:round(0.05 * shape_z), :round(0.05 * shape_y), :] = 1
        mask = numpy.round(mask).astype('uint32')  # binarize
        mask = numpy.repeat(mask, axis=2, repeats=shape_x)
        return mask

    def binarize_pyplot_to_mask(self, tmp_mask_path):
        text_mask_img = SimpleITK.ReadImage(tmp_mask_path)
        text_mask_2d = SimpleITK.GetArrayFromImage(text_mask_img)
        text_mask_2d = text_mask_2d[::-1]  # image is upside down otherwise
        assert numpy.allclose(text_mask_2d[:, :, 3], 255)  # alpha channel
        mask = numpy.max(text_mask_2d[:, :, :3], axis=2)  # take the maximum value of the RGB channels
        mask = mask.astype('float32')
        mask /= 255  # normalize to [0, 1]
        mask = 1 - mask  # invert: show black text in the mask, but not the white background
        return mask

    def export_for_testing(self, results, s_img, base_path: str):
        with open(os.path.join(base_path, 'results.dill'), 'wb') as f:
            dill.dump(results, f)
        with open(os.path.join(base_path, 's_img.dill'), 'wb') as f:
            dill.dump(s_img, f)
        with open(os.path.join(base_path, 'input_img.dill'), 'wb') as f:
            dill.dump(self.img, f)
        with open(os.path.join(base_path, 'pipeline.dill'), 'wb') as f:
            dill.dump(self.pipeline, f)

    def plot_using_pyplot(self, s_img, tmp_mask_path):
        a = SimpleITK.GetArrayFromImage(s_img)
        pixel_coordinates = self.vertebra_coordinates_from_img()  # z, y, x

        tasks = self.pipeline.fnet.load_tasks_for_evaluation()
        vertebra_annotations = {}
        fnet_output = self.img['tool_outputs'][self.pipeline.fnet.name()]
        for vertebra_idx, name in enumerate(fnet_output['names']):
            (patient_id, vertebra_name) = split_name(name)
            vertebra_annotations[vertebra_name] = ''
            assert patient_id == self.img['patient_id']
            for task in tasks:
                if not isinstance(task, BinaryClassificationTask):
                    continue
                y_pred = numpy.array(fnet_output[task.output_layer_name()][vertebra_idx]).item()
                if y_pred > 0.5:
                    vertebra_annotations[vertebra_name] += f'{task.positive_class_name()}: {y_pred:.1%}\n'
                else:
                    vertebra_annotations[vertebra_name] += f'{task.negative_class_name()}: {1 - y_pred:.1%}\n'
            vertebra_annotations[vertebra_name] = vertebra_annotations[vertebra_name].strip()
        results = {
            vertebra_idx: (text, [{'pos': pixel_coordinates[vertebra_name], 'size': 864.07470703125}])
            for vertebra_idx, (vertebra_name, text) in enumerate(vertebra_annotations.items())
        }
        with open(tmp_mask_path.replace('.png', '.json'), 'w') as f:
            json.dump(results, f)
        shape_z, shape_y, shape_x = a.shape
        # Compute DPI from spacing
        spacing = s_img.GetSpacing()  # x, y, z
        mm_per_inch = 25.4
        dpis = tuple(mm_per_inch / s for s in spacing)
        dpi_y = dpis[1]
        dpi_z = dpis[2]
        # pyplot does not allow different dpi values along width and height axes
        # the workaround is to change to plot with fixed dpi first and then resize the output of pyplot later
        img_width_mm = shape_y * spacing[1]
        img_height_mm = shape_z * spacing[2]
        img_width_inches = img_width_mm / mm_per_inch
        img_height_inches = img_height_mm / mm_per_inch
        mask_dpi = max(dpi_y, dpi_z)
        tmp_mask_spacing = (spacing[0], mm_per_inch / mask_dpi, mm_per_inch / mask_dpi)
        # Compute image size in inches
        fig_size = (img_width_inches, img_height_inches)  # 2D sagittal plot for now
        # create a plot using matplotlib or another tool
        pyplot.figure(figsize=fig_size, dpi=mask_dpi)
        # fill the plot with information about vertebral fractures
        img_height_px = img_height_inches * mask_dpi
        img_width_px = img_width_inches * mask_dpi
        pyplot.ylim(0, img_height_px)
        pyplot.xlim(0, img_width_px)
        plot_xy = []
        pyplot.axis('off')
        text_pad_px = {
            'left': 10,
            'right': -10,
        }
        other = {
            'left': 'right',
            'right': 'left',
        }
        ha = 'left'
        for text, pos_info in sorted(results.values(), key=(lambda x: x[1][0]['pos'][0])):
            x, y, z = s_img.TransformPhysicalPointToContinuousIndex(pos_info[0]['pos'][::-1])
            y = y * s_img.GetSpacing()[1] / tmp_mask_spacing[1]
            z = z * s_img.GetSpacing()[2] / tmp_mask_spacing[2]
            s = text
            pyplot.text(y + text_pad_px[ha], z, s, color='black', fontdict={'size': 48}, transform=IdentityTransform(), ha=ha, va='center')
            plot_xy.append((y, z))
            ha = other[ha]  # alternate left and right plotting
        if len(plot_xy) > 0:
            pyplot.scatter(*zip(*plot_xy), color='black', s=64, transform=IdentityTransform(), clip_on=False)
        # save the plot to the results directory
        pyplot.savefig(tmp_mask_path, bbox_inches=0)
        pyplot.close()
        self.export_for_testing(results, s_img, os.path.dirname(tmp_mask_path))

    def nii_to_dcm_seg(self, nii_path: str, base_dcm_dir_path: str, to_dcm_seg_path: str):
        """
        Source https://razorx89.github.io/pydicom-seg/guides/write.html
        """

        dcm_image = self.read_dcm(base_dcm_dir_path)
        dcm_image_metadata = self.read_dcm_metadata(base_dcm_dir_path)

        writer = pydicom_seg.MultiClassWriter(
            template=self.dcm_seg_template(dcm_image_metadata),
            inplane_cropping=False,
            skip_empty_slices=False,
            skip_missing_segment=False,
        )

        segmentation_data = SimpleITK.ReadImage(nii_path)

        dcm = writer.write(segmentation_data, self.dcm_paths(base_dcm_dir_path))
        dcm.save_as(to_dcm_seg_path)
        print('Wrote', os.path.abspath(to_dcm_seg_path))

    def dcm_seg_template(self, dcm_image_metadata: Union[SimpleITK.Image, SimpleITK.ImageFileReader]):
        with open(os.path.join(os.path.dirname(inspect.getfile(HNetFNetPipeline)), 'dcm_seg_template.json')) as f:
            template_dict = json.load(f)
        tag_map = {
            'ContentCreatorName': '0070|0084',
            'ClinicalTrialSeriesID': '0012|0071',
            'ClinicalTrialTimePointID': '0012|0050',
            'SeriesNumber': '0020|0011',
            'InstanceNumber': '0020|0013',
            'ClinicalTrialCoordinatingCenterName': '0012|0060',
            'BodyPartExamined': '0018|0015',
        }
        existing_keys = set(dcm_image_metadata.GetMetaDataKeys())
        for k in list(template_dict.keys()):
            if template_dict[k] is None:
                if tag_map[k] in existing_keys:
                    template_dict[k] = dcm_image_metadata.GetMetaData(tag_map[k])
                else:
                    del template_dict[k]
        template = pydicom_seg.template.from_dcmqi_metainfo(template_dict)
        return template

    def read_dcm(self, base_dcm_dir_path):
        reader = SimpleITK.ImageSeriesReader()
        dcm_files = reader.GetGDCMSeriesFileNames(base_dcm_dir_path)
        reader.SetFileNames(dcm_files)
        image = reader.Execute()
        return image

    def some_dcm_slice(self, base_dcm_dir_path):
        dcm_files = self.dcm_paths(base_dcm_dir_path)
        return dcm_files[0]

    def read_dcm_metadata(self, base_dcm_dir_path):
        dcm_files = self.dcm_paths(base_dcm_dir_path)
        metadata_reader = SimpleITK.ImageFileReader()
        metadata_reader.SetFileName(dcm_files[0])
        metadata_reader.LoadPrivateTagsOn()
        metadata_reader.ReadImageInformation()
        return metadata_reader

    def dcm_paths(self, base_dcm_dir_path):
        return SimpleITK.ImageSeriesReader.GetGDCMSeriesFileNames(base_dcm_dir_path)

    def vertebra_coordinates_from_img(self) -> Dict[str, Tuple[float, float, float]]:
        pixel_coordinates = {}
        for vertebra_name in VERTEBRAE:
            if vertebra_name not in self.img['tool_outputs'][self.pipeline.hnet.name()]:
                continue
            vertebra_data = self.img['tool_outputs'][self.pipeline.hnet.name()][vertebra_name]
            assert len(vertebra_data) == 1, len(vertebra_data)
            vertebra_data = vertebra_data[0]
            if vertebra_data['pos'] is None:
                continue
            pixel_coordinates[vertebra_name] = vertebra_data['pos']  # x,y,z
            pixel_coordinates[vertebra_name] = pixel_coordinates[vertebra_name][::-1]  # z,y,x
        return pixel_coordinates

    def infer_nii_path_if_necessary(self):
        return self.pipeline.infer_nii_path_if_necessary(str(self.img.path))

    def dump_inputs(self):
        with open('input_img.dill', 'wb') as f:
            dill.dump(self.img, f)
        with open('result.dill', 'wb') as f:
            dill.dump(self.result, f)
        with open('pipeline.dill', 'wb') as f:
            dill.dump(self.pipeline, f)

    def rainbow_text(self, x, y, strings, colors, horizontal: bool = True, separate_lines: List[bool] = None, **kw):
        """
        source: https://stackoverflow.com/a/9185851
        """
        if separate_lines is None:
            separate_lines = [False for _ in strings]
        t = pyplot.gca().transData
        fig = pyplot.gcf()
        from matplotlib import transforms
        additional_kwargs = {} if horizontal else {
            'rotation': 90,
            'va': 'bottom',
            'ha': 'center',
        }

        for s, c, l in zip(strings, colors, separate_lines):
            text = pyplot.text(x, y, s + " ", color=c, transform=t, **additional_kwargs, **kw)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            y_offset = horizontal == l
            if y_offset:
                t = transforms.offset_copy(text._transform, y=-ex.height, units='dots')
            else:
                t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


@main_wrapper
def main():
    from model.fnet.data_source import TraumaDataSource
    from model.hnet_fnet.fracture_evaluation import metric_summary
    skip_existing = True
    whole_results_table = []
    for task_name in ['hufo', 'hafo']:
        for test_mode in ['val', 'test']:
            for exclude_c1_c2 in [True, False]:
                for classification_threshold in [0.5, 0.7]:
                    results_subdir = f'{test_mode}_{task_name}_{str(classification_threshold)}'
                    data_source = TraumaDataSource(exclude_c1_c2=exclude_c1_c2)
                    dataset_name_for_evaluation = data_source.name()
                    results_dir = fr'img/generated/{dataset_name_for_evaluation}/trauma_repetition_no_bc_20230314'
                    model_dir = os.path.join('models', 'trauma_repetition_no_bc_20230314')
                    # model_dir = os.path.join(results_dir, 'models')
                    json_output_files = []
                    for model_name in os.listdir(model_dir):
                        if not model_name.endswith('.h5'):
                            continue
                        pipeline = HNetFNetPipeline(
                            hnet=(HNet(model_exe_path=hnet_by_version(7).exe_path,
                                       results_dir=os.path.join(results_dir, 'whole_pipeline'))),
                            fnet=(FNet(model_path=os.path.join(model_dir, model_name),
                                       dataset_name_for_evaluation=dataset_name_for_evaluation,
                                       use_coordinates='should_be_set-by_pipeline_constructor')),
                            results_subdir=results_subdir,
                            whole_pipeline_evaluation_task_name=task_name,
                            classification_threshold=classification_threshold,
                        )
                        json_output_file = os.path.join(pipeline.results_path(), 'fracture_outputs.json')
                        if os.path.isfile(json_output_file) and skip_existing:
                            print('Using existing file instead of evaluating:', json_output_file)
                        else:
                            _, config = FNetParameterEvaluator.load_trained_model_for_evaluation(pipeline.fnet.model_path, only_config=True)
                            test_fold = config['cv_test_fold']
                            if test_mode == 'test':
                                iml = data_source.load_test_dataset_by_fold_id(test_fold)
                            elif test_mode == 'val':
                                iml = data_source.load_fold_by_id(data_source.val_fold_by_test_fold(test_fold))
                            else:
                                raise ValueError(test_mode)
                            pipeline.evaluate(iml)
                            # output_iml = pipeline.predict(iml)
                            # output_iml_path = os.path.join(pipeline.results_path(), 'pipeline_output_example.iml')
                            # output_iml.dump(output_iml_path)
                            # logging.info(f'Wrote {output_iml_path}')
                        assert os.path.isfile(json_output_file)
                        json_output_files.append(json_output_file)
                    metrics = []
                    for j in json_output_files:
                        with open(j, 'r') as f:
                            metrics.append(json.load(f))
                    metrics_df = pandas.DataFrame.from_records(m['patient_metrics'] for m in metrics)
                    summary_data = {}
                    for column in metrics_df.columns:
                        mean, std, conf = metric_summary(column, metrics_df[column])
                        summary_data[column] = {'mean': float(mean), 'std': std, 'conf': conf, 'values': metrics_df[column].tolist()}

                    summary_data['individual_results_files'] = [j for j in json_output_files]
                    summary_data['confusion'] = {}
                    summary_data['confusion']['tp'] = [[patient_id for patient_id, p in m['patients'].items() if p['y_true'] and p['y_pred']] for m in metrics]
                    summary_data['confusion']['fp'] = [[patient_id for patient_id, p in m['patients'].items() if not p['y_true'] and p['y_pred']] for m in metrics]
                    summary_data['confusion']['fn'] = [[patient_id for patient_id, p in m['patients'].items() if p['y_true'] and not p['y_pred']] for m in metrics]
                    summary_data['confusion']['tn'] = [[patient_id for patient_id, p in m['patients'].items() if not p['y_true'] and not p['y_pred']] for m in metrics]

        summary_json_path = os.path.join(base, 'whole_pipeline', test_mode, 'summary.json')
        print('Writing summary to', os.path.abspath(summary_json_path))
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f)


if __name__ == '__main__':
    main()
