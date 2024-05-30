import os
from typing import Optional, Dict, Any

import wandb.wandb_run

from lib.util import deterministic_hash, deterministic_hash_str


class WandBRunInterface:
    def __init__(self,
                 run_name: str,
                 config: Dict[str, Any],
                 project: str = None,
                 api_key: str = None,
                 group: Optional[str] = None, settings=None):
        """
        WandB Terminology:
        - Run=Experiment: A single execution of a script (e.g. a single training run)
        - Project: A collection of runs. Every run belongs to a project.
        - Group: A collection of runs within a project. Not every run needs to be in a group.
        - Config: A dictionary of hyperparameters. Every run has a config.
        """
        if settings is None:
            settings = {'quiet': True}
        if api_key is None:
            api_key: str = self.default_api_key()
        if project is None:
            # default: directory name of the current working directory
            project = os.path.basename(os.getcwd())
        self.api_key = api_key
        self.run_name = run_name
        self.project = project
        self.group = group
        self.config = config
        self._run: Optional[wandb.wandb_run.Run] = None
        self.settings = settings
        self.set_hyperparameter_setting_id()

    @staticmethod
    def default_api_key():
        return os.environ.get('WANDB_API_KEY')

    def get_run(self) -> wandb.wandb_run.Run:
        import wandb
        while self._run is None:
            wandb.login(
                key=self.api_key,
            )
            try:
                self._run = wandb.init(
                    settings=self.settings,
                    project=self.project,
                    name=self.run_name,
                    group=self.group,
                    config=self.config,
                    id=self.run_name,
                    resume='allow',  # "allow" uses the run_id to resume a run if it exists and creates a new run otherwise
                )
            except wandb.errors.CommError as e:
                if 'timed out' in str(e):
                    print('Re-trying to connect to wandb after timeout')
                    continue  # re-try
        return self._run

    def finish(self, exit_code: Optional[int] = None, quiet: Optional[bool] = None) -> None:
        if self._run is None:
            raise RuntimeError
        return self._run.finish(exit_code=exit_code, quiet=quiet)

    def commit(self):
        self.get_run().log({})

    @classmethod
    def from_model_path_and_config(cls, model_path: str, config: Dict[str, Any], experiment_name: str):
        return cls(
            run_name=os.path.basename(model_path),
            config=config,
            group=experiment_name,
        )

    def mark_as_crashed(self):
        self.finish(exit_code=-1)

    def set_hyperparameter_setting_id(self):
        exclude = ['cv_test_fold', 'hyperparameter_setting_id']
        identifier = tuple([self.config[p] for p in sorted(self.config.keys()) if p not in exclude])
        self.config['hyperparameter_setting_id'] = deterministic_hash_str(identifier)

