import inspect
import logging
import os


def start_profiling():
    try:
        import yappi
    except ModuleNotFoundError:
        return
    yappi.set_clock_type("wall")
    print(f'Starting yappi profiler.')
    yappi.start()


def currently_profiling_yappi():
    try:
        import yappi
    except ModuleNotFoundError:
        return False
    return len(yappi.get_func_stats()) > 0


def fix_yappi_crash_before_loading_tensorflow():
    try:
        import yappi
    except ModuleNotFoundError:
        return
    if currently_profiling_yappi():
        logging.info('Stopping yappi profiler before loading tensorflow.')
        yappi.stop()
        from lib.memory_control import MemoryLimiter
        MemoryLimiter.limit_memory_usage()
        yappi.clear_stats()
        yappi.set_clock_type("wall")
        logging.info('Restarting yappi profiler after loading tensorflow.')
        yappi.start()


def profile_wall_time_instead_if_profiling():
    try:
        import yappi
    except ModuleNotFoundError:
        return
    currently_profiling = len(yappi.get_func_stats())
    if currently_profiling and yappi.get_clock_type() != 'wall':
        yappi.stop()
        print('Profiling wall time instead of cpu time.')
        yappi.clear_stats()
        yappi.set_clock_type("wall")
        yappi.start()


def dump_pstats_if_profiling(relating_to_object):
    try:
        import yappi
    except ModuleNotFoundError:
        return
    currently_profiling = len(yappi.get_func_stats())
    if currently_profiling:
        try:
            pstats_file = os.path.join(
                'logs',
                'profiling',
                os.path.normpath(inspect.getfile(relating_to_object)) + '.pstat'
            )
        except AttributeError:
            print('WARNING: unable to set pstat file path for profiling.')
            return
        os.makedirs(os.path.dirname(pstats_file), exist_ok=True)
        yappi.get_func_stats()._save_as_PSTAT(pstats_file)
        print(f'Saved profiling log to {os.path.abspath(pstats_file)}.')


class YappiProfiler():
    def __init__(self, relating_to_object):
        self.relating_to_object = relating_to_object

    def __enter__(self):
        start_profiling()
        profile_wall_time_instead_if_profiling()

    def __exit__(self, exc_type, exc_val, exc_tb):
        dump_pstats_if_profiling(self.relating_to_object)
