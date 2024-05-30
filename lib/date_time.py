import datetime
import logging
import math
from typing import List, Union, Dict, Any, Type

import cachetools

from lib.util import EBC


def get_timestamp(start_day, input_format) -> int:
    if isinstance(start_day, int):
        return start_day
    elif isinstance(start_day, str):
        return int(datetime.datetime.timestamp(datetime.datetime.strptime(start_day, input_format)))
    else:
        return int(datetime.datetime.timestamp(start_day))


class Day(EBC):
    def __init__(self, dt: datetime.date, day_id: int):
        self.day_id = day_id
        self.dt = dt
        self._ymd = self.dt.strftime('%Y-%m-%d')

    @classmethod
    def field_types(cls) -> Dict[str, Type]:
        result = super().field_types()
        result['dt'] = SerializableDateTime
        return result

    def timestamp(self):
        dt = self.dt
        if isinstance(dt, datetime.date):
            dt = datetime.datetime(dt.year, dt.month, dt.day)
        return dt.timestamp()

    @staticmethod
    def from_ts(ts: int, day_id: int):
        return Day(
            day_id=day_id,
            dt=datetime.date.fromtimestamp(ts)
        )

    def day_idx(self):
        return self.day_id - 1

    def datetime_from_time_string(self, time_string: str):
        return self.datetime_from_date_and_time_string(self.dt, time_string)

    @staticmethod
    def datetime_from_date_and_time_string(date: datetime.date, time_string: str):
        if len(time_string) == 5:
            time_string += ':00'
        return datetime.datetime.strptime(f'{date.strftime("%Y-%m-%d")} {time_string}', '%Y-%m-%d %H:%M:%S')

    def ymd(self) -> str:
        return self._ymd

    def weekday(self):
        return self.dt.weekday()

    def weekday_as_german_string(self):
        return {
            'Monday': 'Montag',
            'Tuesday': 'Dienstag',
            'Wednesday': 'Mittwoch',
            'Thursday': 'Donnerstag',
            'Friday': 'Freitag',
            'Saturday': 'Samstag',
            'Sunday': 'Sonntag',
        }[self.dt.strftime('%A')]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError
        return self.day_id == other.day_id and self._ymd == other._ymd


@cachetools.cached(cachetools.LRUCache(maxsize=128))
def time_interval_as_days(start_day, day_count, inputformat='%Y-%m-%d') -> List[Day]:
    '''
    :param start_day: date
    :param day_count:
    :return: list of dates as string
    '''
    start_day = get_timestamp(start_day, inputformat)
    results = [
        Day.from_ts(start_day + 86400 * day_idx, day_id=day_idx + 1)
        for day_idx in range(day_count)
    ]
    return results


def weekday_name(index: int, lan='english'):
    """
    :param index: 0..6
    :return: The weekday as name
    """
    if lan == 'english':
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    else:
        weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
    return weekdays[index]


def get_calendarweek_from_datetime(datetime_obj):
    return datetime_obj.strftime('%V')


def timestamp_from_time_of_day_string(time_string):
    ts = datetime.datetime.timestamp(datetime.datetime.strptime('1970-' + time_string + ' +0000', '%Y-%H:%M %z'))
    assert 0 <= ts <= 24 * 60 * 60
    return ts


class SerializableDateTime(datetime.datetime, EBC):
    def __init__(self, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0,
                    microsecond: int = 0):
        super().__init__(year, month, day, hour, minute, second, microsecond)

    def to_json(self) -> Dict[str, Any]:
        return {
            'type': type(self).__name__,
            'dt_str': self.strftime('%Y-%m-%d %H:%M:%S.%f'),
        }

    def __hash__(self):
        return super().__hash__()

    @staticmethod
    def from_datetime(dt):
        return SerializableDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)

    @staticmethod
    def from_date(dt):
        if isinstance(dt, datetime.datetime):
            return SerializableDateTime.from_datetime(dt)
        return SerializableDateTime(dt.year, dt.month, dt.day)

    @staticmethod
    def from_json(data: Dict[str, Any]):
        cls: Type[SerializableDateTime] = EBC.SUBCLASSES_BY_NAME[data['type']]
        if 'dt_str' in data:
            dt = datetime.datetime.strptime(data['dt_str'], '%Y-%m-%d %H:%M:%S.%f')
            data['year'] = dt.year
            data['month'] = dt.month
            data['day'] = dt.day
            data['hour'] = dt.hour
            data['minute'] = dt.minute
            data['second'] = dt.second
            data['microsecond'] = dt.microsecond
        return cls(data['year'], data['month'], data['day'], data['hour'], data['minute'], data['second'], data['microsecond'], data.get('tzinfo', None))

    def __eq__(self, other):
        if isinstance(other, datetime.date) and not isinstance(other, datetime.datetime):
            return self.date() == other
        return (
                self.year == other.year
                and self.month == other.month
                and self.day == other.day
                and self.hour == other.hour
                and self.minute == other.minute
                and self.second == other.second
                and self.microsecond == other.microsecond
                and self.tzinfo == other.tzinfo
        )

    def __ne__(self, other):
        return not self.__eq__(other)


class SerializableTimeDelta(datetime.timedelta, EBC):
    max: 'SerializableTimeDelta'

    def to_json(self) -> Dict[str, Any]:
        return {
            'type': type(self).__name__,
            'seconds': self.total_seconds(),
        }

    @classmethod
    def from_total_seconds(cls, total_seconds) -> 'SerializableTimeDelta':
        try:
            return SerializableTimeDelta(seconds=total_seconds)
        except OverflowError as e:
            if 'must have magnitude <=' in str(e):
                return cls.max

    def positive_infinite(self):
        return self.days == self.max.days and self.seconds == self.max.seconds and self.microseconds == self.max.microseconds

    def total_seconds(self) -> float:
        if self.positive_infinite():
            return math.inf
        return super().total_seconds()

    @classmethod
    def from_timedelta(cls, timedelta: datetime.timedelta):
        return cls(timedelta.days, timedelta.seconds, timedelta.microseconds)

    @staticmethod
    def from_json(data: Dict[str, Any]):
        cls: Type[SerializableTimeDelta] = EBC.SUBCLASSES_BY_NAME[data['type']]
        return cls(data.get('days', 0.), data['seconds'], data.get('microseconds', 0.))

    def __eq__(self, other):
        return self.total_seconds() == other.total_seconds()


SerializableTimeDelta.max = SerializableTimeDelta.from_timedelta(datetime.timedelta.max)
