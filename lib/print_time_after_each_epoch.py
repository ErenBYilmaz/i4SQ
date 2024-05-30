import datetime
import time

from lib.callbacks import AnyCallback


class PrintTimeAfterEachEpoch(AnyCallback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True
        self.last_epoch_end = None

    def on_epoch_end(self, epoch, logs=None):
        if self.last_epoch_end is None:
            print(datetime.datetime.now().strftime("%H:%M:%S"), 'Epoch', epoch, 'completed')
        else:
            print(datetime.datetime.now().strftime("%H:%M:%S"), 'Epoch', epoch, f'completed within {time.perf_counter() - self.last_epoch_end:.1f}s')
        self.last_epoch_end = time.perf_counter()
