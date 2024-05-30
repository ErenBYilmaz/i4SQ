from lib.my_logger import logging
import os
from subprocess import check_output, PIPE, CalledProcessError

from joblib import Memory

tool_call_cache = Memory('.cache/tool_calls', verbose=1)


@tool_call_cache.cache
def call_tool_with_cache(command):
    return call_tool(command)


def call_tool(command, force_cpu=False, cwd=None, verbose=True) -> str:
    try:
        if verbose:
            logging.info(f'Calling `{" ".join(command)}`...')
        sub_env = os.environ.copy()
        if force_cpu:
            sub_env['CUDA_VISIBLE_DEVICES'] = '-1'
        output: bytes = check_output(command, stderr=PIPE, env=sub_env, cwd=cwd)
        output: str = output.decode('utf-8', errors='ignore')
        return output
    except CalledProcessError as e:
        if e.returncode == 1 and 'ResourceExhaustedError' in str(e.stderr) and not force_cpu:
            logging.info('ResourceExhaustedError on the GPU, re-trying on the CPU...')
            return call_tool(command, force_cpu=True, cwd=cwd)
        else:
            stdout = e.stdout.decode('utf-8', errors='ignore')
            stderr = e.stderr.decode('utf-8', errors='ignore')
            if len(stdout) == 0:
                logging.info('stdout was empty.')
            else:
                logging.info('stdout was: ')
                logging.info(stdout)
            if len(stderr) == 0:
                logging.info('stderr was empty.')
            else:
                logging.info('stderr was: ')
                logging.info(stderr)
            raise