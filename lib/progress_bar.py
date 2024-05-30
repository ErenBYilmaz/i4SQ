import functools
import math
import time
from math import floor
from typing import Iterable, Sized, Iterator


class ProgressBar(Sized, Iterable):
    def __iter__(self) -> Iterator:
        self.check_if_num_steps_defined()
        self.current_iteration = -1  # start counting at the end of the first epoch
        self.current_iterator = iter(self._backend)
        self.start_time = time.perf_counter()
        return self

    def __init__(self,
                 num_steps=None,
                 prefix='',
                 suffix='',
                 line_length=75,
                 empty_char='-',
                 fill_char='#',
                 print_eta=True,
                 print_speed=False,
                 decimals=1):
        self.print_speed = print_speed
        self.decimals = decimals
        self.line_length = line_length
        self.suffix = suffix
        self.empty_char = empty_char
        self.prefix = prefix
        self.fill_char = fill_char
        self.print_eta = print_eta
        self.current_iteration = 0
        self.last_printed_value = None
        self.current_iterator = None
        self.start_time = time.perf_counter()

        try:
            self._backend = range(num_steps)
        except TypeError:
            if isinstance(num_steps, Sized):
                if isinstance(num_steps, Iterable):
                    self._backend = num_steps
                else:
                    self._backend = range(len(num_steps))
            elif num_steps is None:
                self._backend = None
            else:
                raise

        assert num_steps is None or isinstance(self._backend, (Iterable, Sized))

    def set_num_steps(self, num_steps):
        try:
            self._backend = range(num_steps)
        except TypeError:
            if isinstance(num_steps, Sized):
                if isinstance(num_steps, Iterable):
                    self._backend = num_steps
                else:
                    self._backend = range(len(num_steps))
            elif num_steps is None:
                self._backend = None
            else:
                raise

        assert num_steps is None or isinstance(self._backend, (Iterable, Sized))

    def __len__(self):
        return len(self._backend)

    def __next__(self):
        self.print_progress()
        try:
            result = next(self.current_iterator)
            self.increment_iteration()
            self.print_progress()
            return result
        except StopIteration:
            self.increment_iteration()
            self.print_progress()
            raise

    def step(self, num_iterations=1):
        self.current_iteration += num_iterations
        self.print_progress()

    def print_progress(self, iteration=None):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Optional  : current iteration (Int)
        """
        if iteration is not None:
            self.current_iteration = iteration
        try:
            progress = self.current_iteration / len(self)
        except ZeroDivisionError:
            progress = 1
        if self.current_iteration == 0:
            self.start_time = time.perf_counter()
        time_spent = (time.perf_counter() - self.start_time)
        if self.print_eta and progress > 0:
            eta = time_spent / progress * (1 - progress)
            if progress == 1:
                eta = f' T = {int(time_spent / 60):02d}:{round(time_spent % 60):02d}'
            else:
                eta = f' ETA {int(eta / 60):02d}:{round(eta % 60):02d}'
        else:
            eta = ''
        if self.print_speed and self.current_iteration:
            spd = self.current_iteration / time_spent
            if spd > 1:
                speed = f'{spd:.3g} i/s'
            else:
                speed = f'{1 / spd:.3g} s/i'
        else:
            speed = ''
        percent = ("{0:" + str(4 + self.decimals) + "." + str(self.decimals) + "f}").format(100 * progress)
        bar_length = self.line_length - len(self.prefix) - len(speed) - len(self.suffix) - len(eta) - 4 - 6
        try:
            filled_length = int(bar_length * self.current_iteration // len(self))
        except ZeroDivisionError:
            filled_length = bar_length
        if not math.isclose(bar_length * progress, round(bar_length * progress)):
            overflow = bar_length * progress - filled_length
            overflow *= 10
            overflow = floor(overflow)
            assert overflow in range(10)
        else:
            overflow = 0
        if overflow > 0:
            bar = self.fill_char * filled_length + str(overflow) + self.empty_char * (bar_length - filled_length - 1)
        else:
            bar = self.fill_char * filled_length + self.empty_char * (bar_length - filled_length)

        print_value = '\r{0} |{1}| {2}% {4}{3}{5}'.format(self.prefix, bar, percent, speed, self.suffix, eta)
        if self.current_iteration == len(self):
            print_value += '\n'  # Print New Line on Complete
        if self.last_printed_value == print_value:
            return
        self.last_printed_value = print_value
        print(print_value, end='')

    def increment_iteration(self):
        self.current_iteration += 1
        if self.current_iteration > len(self):  # catches the special case at the end of the bar
            self.current_iteration %= len(self)

    def monitor(self, func=None):
        """ Decorates the given function func to print a progress bar before and after each call. """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return functools.partial(self.monitor)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.check_if_num_steps_defined()
            self.print_progress()
            result = func(*args, **kwargs)
            self.increment_iteration()
            self.print_progress()
            return result

        return wrapper

    def check_if_num_steps_defined(self):
        if self._backend is None:
            raise RuntimeError('You need to specify the number of iterations before starting to iterate. '
                               'You can either pass it to the constructor or use the method `set_num_steps`.')
