from lib.util import listdir_fullpath
from model.ensemble.fnet_ensemble import FNetEnsembleBuilder
from model.fnet.fnet import FNet
from model.hnet_fnet.hnet_fnet import HNetFNetPipeline
from model.pbl.pbl import PBL

PBL_MODEL_PATH = r'./models/pbl/model_2020_01_23.pbl'
FNET_MODELS_PATH = r'./models/fnet'


def pipeline() -> HNetFNetPipeline:
    some_fnet_model = [p for p in listdir_fullpath(FNET_MODELS_PATH) if p.endswith('.h5')][0]
    return HNetFNetPipeline(
        hnet=PBL(
            model_exe_path=PBL_MODEL_PATH,
            model_version=8,
        ),
        fnet=FNet(
            model_path=some_fnet_model,
        )
    )
