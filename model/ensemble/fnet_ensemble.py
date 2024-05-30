import json
import logging
import os
from typing import Optional

import tensorflow

from lib.util import EBC


class FNetEnsembleBuilder(EBC):
    def __init__(self):
        self._model: Optional[tensorflow.keras.Model] = None
        self._config: Optional[dict] = None

    def name(self) -> str:
        raise NotImplementedError('Abstract method')

    def as_tf_model(self) -> tensorflow.keras.Model:
        raise NotImplementedError('Abstract method')

    def config_dict(self) -> dict:
        raise NotImplementedError('Abstract method')

    def cached_tf_model(self) -> tensorflow.keras.Model:
        if self._model is None:
            self._model = self.as_tf_model()
        return self._model

    def cached_config_dict(self) -> dict:
        if self._config is None:
            self._config = self.config_dict()
        return self._config

    def save(self, dir_path: str = 'models/ensemble', overwrite: bool = False) -> str:
        os.makedirs(dir_path, exist_ok=True)
        model = self.as_tf_model()
        name_without_extension = os.path.join(dir_path, self.name())
        if not overwrite:
            for extension in ['.h5', '.json', '.png']:
                if os.path.exists(name_without_extension + extension):
                    raise FileExistsError('File exists: ' + name_without_extension + extension)
        logging.info(f'Writing model ensemble to {name_without_extension}.*')
        model.save(name_without_extension + '.h5')
        tensorflow.keras.utils.plot_model(model,
                                          to_file=name_without_extension + '.png',
                                          show_shapes=True,
                                          expand_nested=True)
        with open(name_without_extension + '.json', 'w') as f:
            json.dump(self.config_dict(), f, indent=4)
        for extension in ['.h5', '.json', '.png']:
            assert os.path.isfile(name_without_extension + extension)
        return name_without_extension + '.h5'
