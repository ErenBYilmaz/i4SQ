import re
from typing import Union

import pandas

from lib.tuned_cache import TunedMemory

table_data_cache = TunedMemory(location='./.cache/excel_data', verbose=0)


class PatientNotFoundError(KeyError):
    pass


def excel_data(filename, patient_number, header=0):
    patient_number = str(patient_number)
    df = excel_panda(filename, header=header)
    result = df[df['Patients Name'].astype(str) == patient_number]
    if len(result) == 0:
        raise PatientNotFoundError(patient_number, filename)
    return result


def excel_gender(filename, patient_number: Union[int, str], header=0):
    patient_number = str(patient_number)
    df = excel_panda(filename, header=header)
    result = df[df['PatID'].astype(str) == patient_number]
    if len(result) == 0:
        raise PatientNotFoundError(patient_number, filename)
    return result['Gender'].item()


@table_data_cache.cache
def excel_panda(filename, header=0):
    df = pandas.read_excel(filename,
                           sheet_name=0,  # first sheet
                           header=header)
    return df


@table_data_cache.cache
def csv_data(filename, patient_number, header=0):
    df_idx = int(re.fullmatch(r'X(\d\d\d)', patient_number).group(1)) - 1
    return csv_panda(filename, header=header).iloc[df_idx]


@table_data_cache.cache
def csv_panda(filename, header=0):
    df = pandas.read_csv(filename,
                         header=header)
    return df
