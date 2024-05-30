import json
import os.path
import pickle
import tempfile

import pandas
import wandb.wandb_run
from matplotlib import pyplot

from lib.my_logger import logging
from model.fnet.const import EXPERIMENT_NAME


class EvaluationResultSerializer:
    def isfile(self, relative_path):
        raise NotImplementedError('Abstract method')

    def create_directory_if_not_exists(self, relative_path):
        raise NotImplementedError('Abstract method')

    def save_json(self, relative_path: str, data: dict):
        raise NotImplementedError('Abstract method')

    def save_current_pyplot_figure(self, relative_path):
        raise NotImplementedError('Abstract method')

    def save_text(self, relative_path: str, text: str):
        raise NotImplementedError('Abstract method')

    def upsert(self):
        raise NotImplementedError('Abstract method')

    def commit(self):
        raise NotImplementedError('Abstract method')

    def save_pkl(self, relative_path: str, obj):
        raise NotImplementedError('Abstract method')

    def save_dataframe(self, relative_path: str, df: pandas.DataFrame):
        raise NotImplementedError('Abstract method')


class FileSystemSerializer(EvaluationResultSerializer):
    def isfile(self, relative_path):
        return os.path.isfile(relative_path)

    def create_directory_if_not_exists(self, relative_path):
        os.makedirs(relative_path, exist_ok=True)

    def save_json(self, relative_path: str, data: dict):
        logging.info('Writing ' + relative_path)
        with open(relative_path, 'w') as f:
            json.dump(data, f, indent=4)

    def save_current_pyplot_figure(self, relative_path):
        pyplot.savefig(relative_path)

    def upsert(self):
        pass  # Everything is saved and synchronized immediately

    def commit(self):
        pass  # Everything is saved and synchronized immediately

    def save_text(self, relative_path: str, text: str):
        logging.info('Writing ' + relative_path)
        with open(relative_path, 'w') as f:
            f.write(text)

    def save_pkl(self, relative_path: str, obj):
        logging.info('Writing ' + relative_path)
        with open(relative_path, 'wb') as f:
            pickle.dump(obj, f)

    def save_dataframe(self, relative_path: str, df: pandas.DataFrame):
        logging.info('Writing ' + relative_path)
        df.to_csv(relative_path, index=False)


class WandBSerializer(EvaluationResultSerializer):
    def __init__(self,
                 group_name: str = None,
                 project: str = None,
                 api_key: str = None,
                 settings=None, ):
        """
        See also: https://docs.wandb.ai/guides/artifacts/construct-an-artifact
        """
        if settings is None:
            settings = {'quiet': True}
        if api_key is None:
            api_key: str = os.environ.get('WANDB_API_KEY')
        if project is None:
            # default: directory name of the current working directory
            project = os.path.basename(os.getcwd())
        if group_name is None:
            group_name = EXPERIMENT_NAME
        self.project = project
        self.group = group_name
        wandb.login(key=api_key)
        self._run: wandb.wandb_run.Run = wandb.init(group=group_name, project=project, settings=settings)
        self.artifact = wandb.Artifact(
            name='evaluation_results',
            type='dataset',
        )
        self.settings = settings

    def isfile(self, relative_path):
        raise NotImplementedError('TODO')

    def create_directory_if_not_exists(self, relative_path):
        pass  # wandb takes care of this

    def save_json(self, relative_path: str, data: dict):
        logging.info('Writing ' + relative_path)
        tmp_file = self.create_temporary_file()
        with tmp_file as f:
            json.dump(data, f, indent=4)
        self.artifact.add_file(f.name, name=relative_path)

    def save_current_pyplot_figure(self, relative_path):
        raise NotImplementedError('TODO')

    @staticmethod
    def create_temporary_file():
        return tempfile.NamedTemporaryFile()

    def upsert(self):
        self._run.upsert_artifact(self.artifact)

    def commit(self):
        self._run.finish_artifact(self.artifact)

    def save_text(self, relative_path: str, text: str):
        logging.info('Writing ' + relative_path)
        tmp_file = self.create_temporary_file()
        with tmp_file as f:
            f.write(text)
        self.artifact.add_file(f.name, name=relative_path)

    def save_pkl(self, relative_path: str, obj):
        logging.info('Writing ' + relative_path)
        tmp_file = self.create_temporary_file()
        with tmp_file as f:
            pickle.dump(obj, f)
        self.artifact.add_file(f.name, name=relative_path)

    def save_dataframe(self, relative_path: str, df: pandas.DataFrame):
        table = wandb.Table(dataframe=df)
        self.artifact.add(table, name=relative_path)
