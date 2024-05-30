import datetime
import gc
import logging
import os
import re
import subprocess
import time
from collections import OrderedDict
from math import inf
from typing import Union
from time import sleep

import psutil

try:
    from config import USE_MEMORY_CONTROL
except ImportError:
    USE_MEMORY_CONTROL = False
try:
    from config import GPU_MEMORY_USAGE
except ImportError:
    GPU_MEMORY_USAGE: Union[float, str] = 'growth'
try:
    from config import REVERSE_GPU_ORDER
except ImportError:
    REVERSE_GPU_ORDER: bool = True
try:
    from config import TORCH_ALLOCATE_MEMORY_IN_ADVANCE
except ImportError:
    TORCH_ALLOCATE_MEMORY_IN_ADVANCE: bool = False


class NVLog(dict):
    __indent_re__ = re.compile('^ *')
    __version_re__ = re.compile(r'v([0-9.]+)$')

    def __init__(self):
        super().__init__()

        lines = run_cmd(['nvidia-smi', '-q'])
        lines = lines.splitlines()
        while '' in lines:
            lines.remove('')

        path = [self]
        self['version'] = self.__version__()
        for line in lines[1:]:
            indent = NVLog.__get_indent__(line)
            line = NVLog.__parse_key_value_pair__(line)
            while indent < len(path) * 4 - 4:
                path.pop()
            cursor = path[-1]
            if len(line) == 1:
                if line[0] == 'Processes':
                    cursor[line[0]] = []
                else:
                    cursor[line[0]] = {}
                cursor = cursor[line[0]]
                path.append(cursor)
            elif len(line) == 2:
                if line[0] in ['GPU instance ID', 'Compute instance ID']:
                    continue
                if line[0] == 'Process ID':
                    cursor.append({})
                    cursor = cursor[-1]
                    path.append(cursor)
                cursor[line[0]] = line[1]

        self['Attached GPUs'] = OrderedDict()
        keys = list(self.keys())
        for i in keys:
            if i.startswith('GPU '):
                self['Attached GPUs'][i] = self[i]
                del self[i]

    @staticmethod
    def __get_indent__(line):
        return len(NVLog.__indent_re__.match(line).group())

    @staticmethod
    def __parse_key_value_pair__(line):
        result = line.split(' : ')
        result[0] = result[0].strip()
        if len(result) > 1:
            try:
                result[1] = int(result[1])
            except:
                pass
            if result[1] in ['N/A', 'None']:
                result[1] = None
            if result[1] in ['Disabled', 'No']:
                result[1] = False
        return result

    def __get_processes__(self):
        processes = []
        for i, gpu in enumerate(self['Attached GPUs']):
            gpu = self['Attached GPUs'][gpu]
            if gpu['Processes']:
                for j in gpu['Processes']:
                    processes.append((i, j))
        return processes

    @staticmethod
    def __version__():
        lines = run_cmd(['nvidia-smi', '-h'])
        lines = lines.splitlines()
        result = NVLog.__version_re__.search(lines[0]).group(1)
        return result

    def gpu_table(self):
        output = []
        output.append(self['Timestamp'])
        output.append('+-----------------------------------------------------------------------------+')
        values = []
        values.append(self['version'])
        values.append(self['Driver Version'])
        if 'CUDA Version' in self:
            values.append(self['CUDA Version'])
        else:
            values.append('N/A')
        output.append('| NVIDIA-SMI %s       Driver Version: %s       CUDA Version: %-5s    |' % tuple(values))
        output.append('|-------------------------------+----------------------+----------------------+')
        output.append('| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |')
        output.append('| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |')
        output.append('|===============================+======================+======================|')
        for i, gpu in enumerate(self['Attached GPUs']):
            gpu = self['Attached GPUs'][gpu]
            values = []
            values.append(i)
            values.append(gpu['Product Name'])
            values.append('On' if gpu['Persistence Mode'] else 'Off')
            values.append(gpu['PCI']['Bus Id'])
            values.append('On' if gpu['Display Active'] else 'Off')
            output.append('|   %d  %-19s %3s  | %s %3s |                  N/A |' % tuple(values))
            values = []
            values.append(gpu['Fan Speed'].replace(' ', ''))
            values.append(gpu['Temperature']['GPU Current Temp'].replace(' ', ''))
            values.append(gpu['Performance State'])
            values.append(int(float(gpu['Power Readings']['Power Draw'][:-2])))
            values.append(int(float(gpu['Power Readings']['Power Limit'][:-2])))
            values.append(gpu['FB Memory Usage']['Used'].replace(' ', ''))
            values.append(gpu['FB Memory Usage']['Total'].replace(' ', ''))
            values.append(gpu['Utilization']['Gpu'].replace(' ', ''))
            values.append(gpu['Compute Mode'])
            output.append('| %3s   %3s    %s   %3dW / %3dW |  %8s / %8s |    %4s     %8s |' % tuple(values))
            output.append('+-----------------------------------------------------------------------------+')
        return '\n'.join(output)

    def processes_table(self):
        output = []
        output.append('+-----------------------------------------------------------------------------+')
        output.append('| Processes:                                                       GPU Memory |')
        output.append('|  GPU       PID   Type   Process name                             Usage      |')
        output.append('|=============================================================================|')
        processes = self.__get_processes__()
        if len(processes) == 0:
            output.append('|  No running processes found                                                 |')
        for i, process in processes:
            values = []
            values.append(i)
            values.append(process['Process ID'])
            values.append(process['Type'])
            if len(process['Name']) > 42:
                values.append(process['Name'][:39] + '...')
            else:
                values.append(process['Name'])
            values.append(process['Used GPU Memory'].replace(' ', ''))
            output.append('|   %2d     %5d %6s   %-42s %8s |' % tuple(values))
        output.append('+-----------------------------------------------------------------------------+')
        return '\n'.join(output)

    def as_table(self):
        output = []
        output.append(self.gpu_table())
        output.append('')
        output.append(self.processes_table())
        return '\n'.join(output)


class NVLogPlus(NVLog):

    def processes_table(self):
        output = ['+-----------------------------------------------------------------------------+',
                  '| Processes:                                                       GPU Memory |',
                  '|  GPU       PID   User   Process name                             Usage      |',
                  '|=============================================================================|']
        processes = self.__get_processes__()
        if len(processes) == 0:
            output.append('|  No running processes found                                                 |')
        for i, process in processes:
            values = []
            values.append(i)
            values.append(process['Process ID'])
            p = psutil.Process(process['Process ID'])
            with p.oneshot():
                values.append(p.username()[:8].center(8))
                command = p.cmdline()
                command[0] = os.path.basename(command[0])
                command = ' '.join(command)
                if len(command) > 42:
                    values.append(command[:39] + '...')
                else:
                    values.append(command)
            values.append(process['Used GPU Memory'].replace(' ', ''))
            output.append('|   %2d     %5d %8s %-42s %8s |' % tuple(values))
        output.append('+-----------------------------------------------------------------------------+')
        return '\n'.join(output)


def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')


class MemoryLimiter:
    limited = False

    @classmethod
    def limit_memory_usage(cls, verbose=1):
        if not USE_MEMORY_CONTROL:
            print('memory_control module disabled.')
            return
        if cls.limited:
            if verbose:
                print('Already limited memory usage. Skipping...')
            return
        use_gpu_idx = None
        printed_compute_mode = False
        while True:  # until a GPU is free
            try:
                data = NVLog()
            except FileNotFoundError:
                print('WARNING: nvidia-smi is not available')
                break
            if 'use_specific_gpu' in os.environ:
                print(f'use_specific_gpu = {os.environ["use_specific_gpu"]}')
                use_gpu_idx = int(os.environ['use_specific_gpu'])
            else:
                gpus = list(enumerate(data['Attached GPUs'].values()))
                if REVERSE_GPU_ORDER:
                    gpus = reversed(gpus)
                for idx, gpu_data in gpus:
                    any_processes = (
                            gpu_data['Processes'] is not None
                            or gpu_data['Utilization']['Memory'] != '0 %'
                            or gpu_data['FB Memory Usage']['Used'] != '0 MiB'
                    )
                    compute_mode = gpu_data['Compute Mode']
                    if not printed_compute_mode:
                        print('GPU Compute Mode:', compute_mode)
                        printed_compute_mode = True
                    if compute_mode in ['Exclusive_Process', 'Exclusive_Thread'] or os.environ.get('use_empty_gpu'):
                        if not any_processes:
                            use_gpu_idx = idx
                            break
                    elif compute_mode == 'Default':
                        if GPU_MEMORY_USAGE != 'growth':
                            free_memory = int(re.search(r'(\d+) MiB', gpu_data['FB Memory Usage']['Free']).group(1))
                            if free_memory > 2.5 * GPU_MEMORY_USAGE:
                                use_gpu_idx = idx
                                break
                        else:
                            use_gpu_idx = idx
                            break
                    elif compute_mode == 'Prohibited':
                        continue
                    else:
                        raise NotImplementedError(f'Unknown compute mode: {compute_mode}.')
                else:
                    print(datetime.datetime.now().strftime("%H:%M") + ': All GPUs are currently in use.')
                    sleep(300)
                    continue
            os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu_idx)
            print('Using GPU', f'{use_gpu_idx}:', list(data['Attached GPUs'].values())[use_gpu_idx]['Product Name'])
            break

        cls.limit_gpu_memory_usage()

        cls.limit_ram_usage()

        cls.limited = True

    @classmethod
    def limit_ram_usage(cls):
        try:
            from lib.memory_limit_windows import create_job, limit_memory, assign_job
            import psutil

            ram_limit = psutil.virtual_memory().total * 2 // 3
            print('Limiting RAM usage to {0:,} Bytes.'.format(ram_limit))

            assign_job(create_job())
            limit_memory(ram_limit)
        except ModuleNotFoundError:
            try:
                from lib.memory_limit_linux import limit_memory, get_memory

                ram_limit = get_memory() * 2 // 3
                print('Limiting RAM usage to {0:,} Bytes.'.format(ram_limit))

                limit_memory(ram_limit)
            except ModuleNotFoundError:
                print('WARNING: Setting memory limit failed. '
                      'This can happen if you are not on Windows nor on Linux or if you have forgot to install some dependencies.')

    @classmethod
    def limit_gpu_memory_usage(cls):
        import tensorflow as tf
        # gpu_memory_limit = 3.5 * 1024
        for gpu in tf.config.experimental.list_physical_devices('GPU'):  # only 1 should be visible
            if GPU_MEMORY_USAGE == 'growth':
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    # no idea why the 0.75 factor is needed but tensorflow uses more memory than we allow here
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEMORY_USAGE * 0.75)
                ])


class MemoryLimiterTorch(MemoryLimiter):
    @classmethod
    def limit_gpu_memory_usage(cls):
        import torch.cuda
        for gpu in range(torch.cuda.device_count()): # only 1 should be visible
            torch.cuda.set_per_process_memory_fraction(0.95, gpu)  # dont use whole GPU
            max_bytes = GPU_MEMORY_USAGE * 1024 * 1024  # GPU_MEMORY_USAGE is in MB
            if GPU_MEMORY_USAGE == 'growth':
                pass  # default in pytorch
            else:
                torch.cuda.set_device(gpu)
                torch.cuda.empty_cache()
                total_memory = torch.cuda.get_device_properties(gpu).total_memory
                torch.cuda.set_per_process_memory_fraction(max_bytes / total_memory, gpu)
                logging.info(f'Setting max_memory_per_process to {max_bytes / total_memory:.2%}')


def tf_memory_leak_cleanup():
    if time.perf_counter() - tf_memory_leak_cleanup.last_cleanup_ended < 60:
        return  # at most every 60 seconds to unnecessary iterations that can be expensive
    import tensorflow
    for obj in gc.get_objects():
        if isinstance(obj, tensorflow.Graph):
            if hasattr(obj, '_py_funcs_used_in_graph'):
                del obj._py_funcs_used_in_graph[:]
        if isinstance(obj, tensorflow.keras.utils.GeneratorEnqueuer):
            obj.stop()
    gc.collect()
    tf_memory_leak_cleanup.last_cleanup_ended = time.perf_counter()


tf_memory_leak_cleanup.last_cleanup_ended = -inf
