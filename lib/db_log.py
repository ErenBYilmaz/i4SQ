import logging
import os
import sqlite3 as db
import sys
from math import inf
from shutil import copyfile
import time
from typing import Optional
import git

DBName = str

connected_dbs = [DBName]

# get current commit at start time of program
repo = git.Repo(search_parent_directories=True)
CURRENT_SHA = repo.head.object.hexsha


class DBLog:
    def __init__(self, db_name='log.db', create_if_not_exists=True):
        if db_name in connected_dbs:
            raise ValueError(f'There is already a connection to {db_name}.'
                             'If you want to re-use the same connection you can get it from `db_log.connected_dbs`.'
                             'If you want to disconnect you can call log.disconnect().')
        self.connection: Optional[db.Connection] = None
        self.cursor: Optional[db.Cursor] = None
        self.db_name: Optional[DBName] = None

        db_name = db_name.lower()
        if not os.path.isfile(db_name) and not create_if_not_exists:
            raise FileNotFoundError('There is no database with this name.')
        creating_new_db = not os.path.isfile(db_name)
        try:
            db_connection = db.connect(db_name, check_same_thread=False)
            # db_setup.create_functions(db_connection)
            # db_setup.set_pragmas(db_connection.cursor())
            # connection.text_factory = lambda x: x.encode('latin-1')
        except db.Error as e:
            print("Database error %s:" % e.args[0])
            raise

        self.connection = db_connection
        self.cursor = self.connection.cursor()
        self.db_name = db_name
        if creating_new_db:
            try:
                if os.path.isfile('/test-db/' + db_name):
                    print('Using test logs')
                    copyfile('/test-db/' + db_name, db_name)
                else:
                    self.setup()
            except Exception:
                if self.connection is not None:
                    self.connection.rollback()
                os.remove(db_name)
                raise
        self.connected = True
        self.min_level = -inf

    def disconnect(self, rollback=True):
        if rollback:
            self.connection.rollback()
        else:
            self.connection.commit()
        self.connection.close()
        self.connected = False

    def setup(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries(
            rowid INTEGER PRIMARY KEY,
            dt_created DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            message TEXT NOT NULL,
            data BLOB, -- can be null
            pid INTEGER NOT NULl,
            message_type VARCHAR (25) NOT NULL, -- a plain text message title
            level INTEGER NOT NULL, -- relates to logging.ERROR and similar ones
            head_hex_sha VARCHAR-- SHA of currently checked out commit
        )
        ''')

    def log(self,
            message,
            level,
            message_type='generic',
            data=None,
            dt_created=None,
            current_pid=None,
            current_head_hex_sha=CURRENT_SHA,
            data_serialization_method=lambda x: x):
        if level < self.min_level:
            return
        if dt_created is None:
            dt_created = round(time.time())
        if current_pid is None:
            current_pid = os.getpid()
        data: str = data_serialization_method(data)
        self.cursor.execute('''
        INSERT INTO entries(message, data, dt_created, pid, head_hex_sha, message_type, level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (message, data, dt_created, current_pid, current_head_hex_sha, message_type, level))

    def debug(self, message, *args, **kwargs):
        self.log(message, logging.DEBUG, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.log(message, logging.INFO, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.log(message, logging.WARNING, *args, **kwargs)

    warn = warning

    def error(self, message, *args, **kwargs):
        self.log(message, logging.ERROR, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.log(message, logging.CRITICAL, *args, **kwargs)

    fatal = critical

    def exception(self, msg, *args, data=None, **kwargs):
        if data is None:
            data = sys.exc_info()
        self.error(msg, *args, data=data, **kwargs)

    def commit(self):
        c = time.perf_counter()
        self.connection.commit()
        delta = time.perf_counter() - c
        print(f'Committing log files took {delta} seconds')
        return delta
