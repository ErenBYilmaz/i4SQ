import math
import multiprocessing.util
import re

import time

import random
from unittest.mock import patch

import dill
import os

import joblib.externals.loky.process_executor
from joblib import delayed, Parallel
from typing import Callable, List

from lib import memory_control
from lib.memory_control import NVLog
from lib.my_logger import logging


class SerializableJob:
    def __init__(self, callback: Callable, args, kwargs, subdir: str, filename: str = None, run_repeatedly=False):
        self.filename = filename
        self.subdir = subdir
        self.callback = callback
        self.run_repeatedly = run_repeatedly
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        logging.info(f'Starting job {self.filename}')
        self.callback(*self.args, **self.kwargs)
        if not self.run_repeatedly:
            self.remove_serialized_version()
        logging.info(f'Finished job {self.filename}')

    def delayed(self):
        return delayed(self)()

    @staticmethod
    def jobs_dir():
        return 'scheduled_jobs'

    def serialize(self):
        fp = self.filepath()
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def clear_jobs(cls, subdir: str):
        for j in SerializableJob.remaining_jobs(subdir, verbose=False):
            j.remove_serialized_version()

    def filepath(self):
        last_job_name = max(os.listdir(self.my_jobs_dir()))
        last_job_id = re.match(r'(\d+).dill', last_job_name).group(1)
        if self.filename is None:
            job_id = last_job_id + 1
            self.filename = f'{job_id:05d}.dill'
            filepath = self.serialized_path()
            assert not os.path.isfile(filepath)
        else:
            filepath = self.serialized_path()
        return filepath

    def remove_serialized_version(self):
        try:
            os.remove(self.serialized_path())
        except FileNotFoundError:
            logging.error(f'Could not remove serialized job {self.serialized_path()}')

    def serialized_path(self):
        return os.path.join(self.my_jobs_dir(), self.filename)

    def my_jobs_dir(self):
        return os.path.join(self.jobs_dir(), self.subdir)

    @classmethod
    def remaining_jobs(cls, subdir: str, verbose=True, shuffle=False) -> List['SerializableJob']:
        results = []
        dirpath = os.path.join(cls.jobs_dir(), subdir)
        if os.path.isdir(dirpath):
            for file_name in sorted(os.listdir(dirpath)):
                filepath = os.path.join(dirpath, file_name)
                with open(filepath, 'rb') as f:
                    job = dill.load(f)
                    assert isinstance(job, SerializableJob)
                    job.filename = file_name  # in case someone renamed the job file
                    assert os.path.isfile(job.serialized_path())
                    if verbose and os.path.getmtime(filepath) < time.time() - 60 * 60 * 24 * 7:  # older than 7 days
                        logging.warning('Loading job that is older than 7 days.')
                    results.append(job)
        if verbose and len(results) > 0:
            logging.info(f'Loaded {len(results)} unfinished jobs to be processed.')
        if shuffle:
            random.shuffle(results)
        return results

    @classmethod
    def remaining_jobs_iterator(cls, subdir: str, verbose=True, shuffle=False) -> List['SerializableJob']:
        remaining = cls.remaining_jobs(subdir=subdir, verbose=verbose, shuffle=shuffle)
        while len(remaining) > 0:
            job = remaining.pop(0)
            if job.run_repeatedly:
                remaining.append(job)
            yield job


class MyParallel(Parallel):
    def __init__(self, n_jobs=None, backend=None, verbose=0, timeout=None, pre_dispatch=None, batch_size='auto', temp_folder=None, max_nbytes='1M',
                 mmap_mode='r', prefer=None, require=None, before_dispatch=lambda: None, reuse=False):
        self.before_dispatch = before_dispatch
        self.reuse = reuse
        if pre_dispatch is None:
            if self.reuse:
                pre_dispatch = '2 * n_jobs'
            else:
                pre_dispatch = '1 * n_jobs'
        super().__init__(n_jobs=n_jobs,
                         backend=backend,
                         verbose=verbose,
                         timeout=timeout,
                         pre_dispatch=pre_dispatch,
                         batch_size=batch_size,
                         temp_folder=temp_folder,
                         max_nbytes=max_nbytes,
                         mmap_mode=mmap_mode,
                         prefer=prefer,
                         require=require)
        self.patch_multiprocessing_util_logger()
        if not self.reuse:
            self._backend_args['idle_worker_timeout'] = 0
        self.no_memory_leak_clearing_path = patch('joblib.externals.loky.process_executor._MEMORY_LEAK_CHECK_DELAY', math.inf)

    def patch_multiprocessing_util_logger(self):
        # multiprocessing.util._logger = logging.getLogger()
        multiprocessing.util.info = logging.info
        multiprocessing.util.debug = logging.debug
        print(f'handlers: {multiprocessing.util.get_logger().handlers}')
        joblib.externals.loky.process_executor.mp.util.info('I you read this in the terminal, joblib.externals.loky.process_executor.mp.util.info '
                                                            'was replaced successfully by `my_logger.logging.info`!')
        print('If you dont read the above message referring to multiprocessing.util._logger, it was not properly registered.')
        # joblib.externals.loky.process_executor.mp.util.info = logging.info
        # joblib.externals.loky.process_executor.mp.util.debug = logging.debug

    def __enter__(self):
        if self.reuse:
            self.no_memory_leak_clearing_path.__enter__()
            print('Disabled joblib\'s memory leak cleanup.')
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.reuse:
            self.no_memory_leak_clearing_path.__exit__(exc_type, exc_val, exc_tb)
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _dispatch(self, batch):
        if self._aborting:
            return
        self.before_dispatch()
        if not self.reuse and not self.any_free_workers() and not self._aborting:
            print('DEBUG INFO: With timeout=0 joblib will shut down idle workers, but we still need to make sure that they actually idle '
                  "by not submitting too many jobs. Actually, I think this code should be unreachable because I also set pre_dispatch = '1 * n_jobs'. "
                  'Waiting for a worker job to finish...')
            while True:
                if self._aborting or self.any_free_workers():
                    break
                time.sleep(0.1)
        if self._aborting:
            return
        super()._dispatch(batch)
        print('Completed tasks:', self.n_completed_tasks, 'Dispatched tasks:', self.n_dispatched_tasks)

    def any_free_workers(self):
        tasks_in_running = self.n_dispatched_tasks - self.n_completed_tasks
        any_free_workers = tasks_in_running < self.n_jobs
        return any_free_workers


class BeforeDispatch:
    def __init__(self,
                 only_first_n_jobs: int = None,
                 max_wait_time: float = 120.,
                 sleep_interval=5.,
                 fixed_wait_time: float = None,
                 wait_until_memory_increased_by=None):
        if wait_until_memory_increased_by is None:
            if memory_control.GPU_MEMORY_USAGE != 'growth':
                wait_until_memory_increased_by = memory_control.GPU_MEMORY_USAGE * 0.5
        self.wait_until_memory_increased_by = wait_until_memory_increased_by
        self.sleep_interval = sleep_interval
        self._skip_next = True
        self.max_wait_time = max_wait_time
        self.only_first_n_jobs = only_first_n_jobs
        self.n_dispatched = 0
        self.fixed_wait_time = fixed_wait_time
        self.last_dispatch_time = None

    def get_memory(self):
        data = NVLog()
        return sum(int(re.search(r'(\d+) MiB', gpu_data['FB Memory Usage']['Used']).group(1)) for gpu_data in data['Attached GPUs'].values())

    def before_dispatch(self):
        if self._skip_next:
            self._skip_next = False
            self.on_dispatch()
            return
        if self.fixed_wait_time is not None:
            print(f'Waiting {self.fixed_wait_time} before dispatching next job')
            time.sleep(self.fixed_wait_time)
            self.on_dispatch()
            return
        if self.only_first_n_jobs is not None:
            if self.n_dispatched >= self.only_first_n_jobs:
                self.on_dispatch()
                return  # for example if workers are reused and the memory is already allocated
        start = time.perf_counter() if self.last_dispatch_time is None else self.last_dispatch_time
        memory_used_before = total_memory_used = self.get_memory()
        time.sleep(self.sleep_interval)
        print(f'Waiting for GPU memory usage to change...')
        # The idea is to wait until the previous job allocated its gpu memory (at least 0.75 the planned memory limit)
        # but it could also happen hat a job terminates early or gpu memory is freed, then we also dispatch
        minimum_memory_per_job = self.wait_until_memory_increased_by
        if minimum_memory_per_job is not None:
            lower_limit = memory_used_before - minimum_memory_per_job
            upper_limit = memory_used_before + minimum_memory_per_job
            while lower_limit < total_memory_used < upper_limit:  # MiB
                t = time.perf_counter() - start
                print(f'Memory at {total_memory_used} MiB (compared to {lower_limit:.0f}--{upper_limit:.0f}, t={t :.1f}/{self.max_wait_time:.1f}s)')
                time.sleep(max(min(self.sleep_interval, self.max_wait_time - t), 0))
                total_memory_used = self.get_memory()
                if t >= self.max_wait_time:
                    print(f'Max wait time reached.')
                    break
            time.sleep(self.sleep_interval)
        self.n_dispatched += 1
        self.on_dispatch()

    def on_dispatch(self):
        logging.info(f'Dispatching job at GPU memory usage of {self.get_memory()} MiB.')
        self.last_dispatch_time = time.perf_counter()
