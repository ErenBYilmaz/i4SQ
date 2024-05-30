import copy
import math
import os.path

from lib.image_processing_tool import TrainableImageProcessingTool
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer


class StoreMCDOModels(ModelAnalyzer):
    def analyze_batch(self, batch, y_preds, names):
        raise NotImplementedError('Not needed for this analyzer. Do not call.')

    def __init__(self, evaluator: FNetParameterEvaluator, skip_existing=None):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False, skip_existing=skip_existing, model_level_caching=False,
                         binary_threshold_or_method=0.5)

    def to_subdir(self):
        return 'models_mcdo'

    def compute_classification_threshold_for_model(self, task_idx: int, _cache_key) -> float:
        return 0.5

    def analyze_single_model(self, model_path: str, ) -> SingleModelEvaluationResult:
        self.serializer().create_directory_if_not_exists(self.to_dir())
        filename = os.path.basename(model_path)
        assert filename.endswith('.h5')
        filename = filename[:-len('.h5')] + '_mcdo.h5'
        to_file = os.path.join(self.to_dir(), filename)
        if self.skip_existing and self.serializer().isfile(to_file):
            return []
        self.before_model(model_path)
        assert self.ONLY_TASKS is None
        model = self.load_trained_model_for_evaluation(include_artificial_outputs=False)
        base_config = model.config
        if math.isclose(base_config['dropout_rate'], 0):
            with open(os.path.join(os.path.dirname(to_file), 'where_to_find.txt'), 'a') as f:
                f.write(f'Model {os.path.basename(model_path)} has no Dropout rate: No MCDO possible.\n')
            return []
        if 'monte_carlo_dropout' in base_config and base_config['monte_carlo_dropout']:
            with open(os.path.join(os.path.dirname(to_file), 'where_to_find.txt'), 'a') as f:
                f.write(f'Model {os.path.basename(model_path)} already uses MCDO. See `models` directory.\n')
            return []
        mcdo_model = model.to_mcdo_model()
        self.save_mcdo_model(mcdo_model, to_file)
        return []

    def save_mcdo_model(self, mcdo_model, to_file):
        assert to_file.endswith('.h5')
        mcdo_model.model_path = to_file[:-len('.h5')]
        mcdo_model.save_model_and_config()
        assert os.path.isfile(mcdo_model.model_path + '.h5')
        assert os.path.isfile(mcdo_model.model_path + '.json')
        print(f'Wrote {os.path.abspath(to_file)}')
