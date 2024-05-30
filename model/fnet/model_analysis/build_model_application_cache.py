from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer


class BuildModelApplicationCache(ModelAnalyzer):
    def __init__(self, evaluator: FNetParameterEvaluator,
                 include_xs: bool = False):
        super().__init__(evaluator=evaluator,
                         model_level_caching=True,
                         analysis_needs_xs=include_xs)

    def to_subdir(self):
        return 'cache'

    def analyze_batch(self, batch, y_preds, names):
        pass

    @staticmethod
    def both(evaluator: FNetParameterEvaluator):
        return BuildModelApplicationCache(evaluator, include_xs=True), BuildModelApplicationCache(evaluator, include_xs=False)
