from __future__ import print_function

import sys
import threading
from functools import partial
from time import sleep

try:
    import thread
except ImportError:
    import _thread as thread

try:  # use code that works the same in Python 2 and 3
    range, _print = xrange, print


    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        _print(*args, **kwargs)
        if flush:
            kwargs.get('file', sys.stdout).flush()
except NameError:
    pass


class Cancellation:
    def __init__(self, canceled=False):
        self.canceled = canceled


def cdquit(fn_name, cancellation):
    if cancellation.canceled:
        return
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''

    def outer(fn):
        def inner(*args, **kwargs):
            c = Cancellation()
            timer = threading.Timer(s, partial(cdquit, cancellation=c), args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                c.canceled = True
                timer.cancel()
            return result

        return inner

    return outer


def call_method_with_timeout(method, timeout, *args, **kwargs):
    return exit_after(timeout)(method)(*args, **kwargs)


