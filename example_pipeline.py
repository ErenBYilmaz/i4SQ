from model.hnet_fnet.hnet_fnet import HNetFNetPipeline

PBL_MODEL_PATH = r'./models/pbl/model_2020_01_23.pbl'
FNET_MODELS_PATH = r'./models/fnet'

def pipeline():
    return HNetFNetPipeline(
        hnet=Pbl()
    )