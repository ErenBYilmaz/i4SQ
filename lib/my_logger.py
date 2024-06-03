import os
import logging
import sys

# noinspection PyUnresolvedReferences

logging = logging
format = "%(asctime)-15s %(levelname)-8s %(message)s"
handlers = []
try:
    import __main__
    logfile = 'logs/' + os.path.normpath(__main__.__file__).replace(os.path.abspath('.'), '') + '.log'
except (AttributeError, ModuleNotFoundError):
    print('WARNING: unable to set log file path.')
else:
    try:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        file_logger = logging.FileHandler(logfile, mode="a+", encoding="UTF-8")
        handlers.append(file_logger)
    except OSError:
        print('WARNING: unable to set log file path.')
    del __main__

stdout_logger = logging.StreamHandler(sys.stdout)
stdout_logger.setFormatter(logging.Formatter(format))
handlers.append(stdout_logger)
logging.basicConfig(level=logging.INFO, handlers=handlers, format=format)
