import functools
import inspect
import os
import time

from typing import List

from lib.my_logger import logging
from lib.print_exc_plus import print_exc_plus
from lib.profiling_tools import start_profiling, profile_wall_time_instead_if_profiling, dump_pstats_if_profiling

try:
    from config import SKIP_STACK_DUMP
except ImportError:
    SKIP_STACK_DUMP = False
try:
    import winsound as win_sound


    def beep(*args, **kwargs):
        win_sound.Beep(*args, **kwargs)
except ImportError:
    win_sound = None
    beep = lambda x, y: ...

ENABLE_PROFILING = False


def list_logger(base_logging_function, store_in_list: list):
    def print_and_store(*args, **kwargs):
        base_logging_function(*args, **kwargs)
        store_in_list.extend(args)

    return print_and_store


EMAIL_CRASHES_TO = []



class MainWrapper:
    def __init__(self,
                 email_crashes_to: List[str] = None,
                 voice_call_on_crash_to: List[str] = None,
                 beep = True,
                 re_raise_unkown_exceptions = False,
                 stack_tracing=False,
                 skip_memory_limit=False):
        if email_crashes_to is None:
            self.email_crashes_to = []
        if voice_call_on_crash_to is None:
            self.voice_call_on_crash_to = []
        self.beep = beep
        self.re_raise_unkown_exceptions = re_raise_unkown_exceptions
        self.stack_tracing = stack_tracing
        self.skip_memory_limit = skip_memory_limit

    def __call__(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not self.skip_memory_limit:
                from lib import dl_backend
                dl_backend.b().limit_memory_usage()
            if ENABLE_PROFILING:
                start_profiling()
            start = time.perf_counter()
            try:
                path = os.path.abspath(inspect.getfile(f))
            except TypeError:
                import __main__
                path = __main__.__file__
            if hasattr(f, '__name__'):
                path += f'.{f.__name__}'
            logging.info(f'Starting {path} Process id: {os.getpid()}')
            # does not help much
            # monitoring_thread = hanging_threads.start_monitoring(seconds_frozen=180, test_interval=1000)
            os.makedirs('logs', exist_ok=True)
            if self.stack_tracing:
                import lib.stack_tracer
                log_to_file = 'logs/' + os.path.split(path)[-1] + '.html'
                interval = 5
                logging.info(f'Stack trace is written every {interval} seconds into {log_to_file}')
                lib.stack_tracer.trace_start(log_to_file, interval=interval)
            # faulthandler.enable()
            profile_wall_time_instead_if_profiling()

            # noinspection PyBroadException
            if SKIP_STACK_DUMP:
                serialize_stack_on_error_to = None
            else:
                serialize_stack_on_error_to = 'logs/' + os.path.split(path)[-1] + '.dill'
            try:
                return f(*args, **kwargs)
            except KeyboardInterrupt:
                error_messages = []
                if os.path.isfile('print_exc_plus_on_keyboard_interrupt.msg'):
                    print_exc_plus(print=list_logger(logging.error, error_messages),
                                   serialize_to=serialize_stack_on_error_to)
            except Exception:
                error_messages = []
                print_exc_plus(print=list_logger(logging.error, error_messages),
                               serialize_to=serialize_stack_on_error_to)
                for recipient in EMAIL_CRASHES_TO:
                    from jobs.sending_emails import send_mail
                    logging.info(f'Sending a mail to {recipient} to notify about the crash.')
                    send_mail.create_simple_mail_via_gmail(body='\n'.join(error_messages), filepath=None, excel_name=None, to_mail=recipient, subject='[python] Crash report')
                for to_number in self.voice_call_on_crash_to:
                    logging.info(f'Calling {to_number} to notify about the crash.')
                    voice_call('This is a notification message that one of your python scripts has crashed. If you are unsure about the origin of this call, please contact Eren Yilmaz.',
                               to_number)
                if self.re_raise_unkown_exceptions:
                    raise
            finally:
                logging.info('Terminated.')
                total_time = time.perf_counter() - start
                # faulthandler.disable()
                # stack_tracer.trace_stop()
                if self.beep:
                    frequency = 2000
                    duration = 500
                    beep(frequency, duration)
                if ENABLE_PROFILING:
                    dump_pstats_if_profiling(f)
                print('Total time', total_time)
            # if unexpected_error:
            #     print(f'Trying to kill parent process with id {os.getppid()}')
            #     if os.getppid() != 0:
            #         # print(f'Trying to kill parent process with id {os.getppid()}')
            #         os.kill(os.getppid(), signal.SIGKILL)

        wrapper.func = f

        return wrapper


main_wrapper = MainWrapper()