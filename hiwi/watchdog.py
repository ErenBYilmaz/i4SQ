import getpass
import logging
import os
import psutil
import re
import smtplib
import socket
import time
import traceback

from collections import OrderedDict
from datetime import timedelta
from email.message import EmailMessage
from tabulate import tabulate
from typing import List, Optional


log = logging.getLogger(__name__)


class Watchdog:
    """Uses system tools to monitor processes and notify you via e-mail if one
    dies. Looks by default only for processes of the current user.
    """

    def __init__(self, host: str, port: int, to: str, interval: int = 30,
                 user: Optional[str] = None, password: Optional[str] = None,
                 starttls: bool = False, from_: Optional[str] = None,
                 patterns: Optional[List[str]] = None,
                 report_every: Optional[float] = None) -> None:
        """
        Args:
            host: The SMTP server to use.
            port: Port of the smtp server.
            to: The email address to send notifications to.
            interval: Polling interval in seconds.
            user: The user to login with.
            password: Password to use.
            starttls: Whether to use STARTTLS.
            patterns: An optional list of strings to match the process against.
            report_every: Send a report of all currently watched processes
                after this amount of seconds.
        """
        assert interval > 0

        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._starttls = starttls
        self._from = from_ or user
        self._to = to
        self._interval = interval
        self._report_every = report_every

        if patterns is not None:
            patterns = [re.compile(pattern, re.I) for pattern in patterns]

        self._patterns = patterns

        self._from = f'{socket.gethostname()}\'s watchdog <{self._from}>'

        self._process_user = getpass.getuser()
        self._process_id = os.getpid()

    def run(self) -> None:
        try:
            self._run()
        except Exception:
            self._send_mail('Watchdog died!', traceback.format_exc())

    def _run(self) -> None:
        prev_processes = self._get_processes()

        process_literal = 'process' + ('es' if len(prev_processes) > 1 else '')
        self._send_mail(f'Start watching {len(prev_processes)} ' +
                        str(process_literal),
                        tabulate(list(prev_processes.values()),
                                 tablefmt='plain'))

        last_report = time.time()

        while True:
            current_processes = self._get_processes()

            log.debug('Found %i matching processes', len(current_processes))

            new_processes = [(v[0], v[2]) for p, v in current_processes.items()
                             if p not in prev_processes]
            dead_processes = [v for p, v in prev_processes.items()
                              if p not in current_processes]

            subjects = []
            bodies = []

            if new_processes:
                process_literal = 'process' + ('es' if len(new_processes) > 1
                                               else '')
                subjects.append(f'{len(new_processes)} new {process_literal}')
                bodies.append(f'New {process_literal}:\n' +
                              tabulate(new_processes, tablefmt='plain'))

            if dead_processes:
                process_literal = 'process' + ('es' if len(dead_processes) > 1
                                               else '')
                subjects.append(f'{len(dead_processes)} {process_literal} '
                                'died')
                bodies.append(f'Dead {process_literal}:\n' +
                              tabulate(dead_processes, tablefmt='plain'))

            now = time.time()

            if self._report_every is not None and \
               (now - last_report) > self._report_every:
                process_literal = 'process' + \
                    ('es' if len(current_processes) > 1 else '')
                subjects.append(f'{len(current_processes)} running '
                                f'{process_literal}')
                bodies.append(f'Running {process_literal}:\n' +
                              tabulate(list(current_processes.values()),
                                       tablefmt='plain'))
                last_report = now

            if subjects:
                subject = ', '.join(subjects)
                body = '\n\n'.join(bodies)
                self._send_mail(subject, body)

            prev_processes = current_processes
            time.sleep(self._interval)

    def _get_processes(self):
        """Gets all current matching processes."""
        processes = OrderedDict()

        for process in psutil.process_iter():
            try:
                username = process.username()

                if self._process_user not in username or \
                   self._process_id == process.pid:
                    continue

                runtime = str(timedelta(seconds=time.time() -
                                        process.create_time()))
                cmdline = ' '.join(process.cmdline())

                if self._patterns is not None and \
                   all(pattern.search(cmdline) is None for pattern in
                       self._patterns):
                    continue

                processes[process] = (process.pid, runtime, cmdline)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

        return processes

    def _send_mail(self, subject: str, body: str) -> None:
        """Sends an email."""
        log.debug('Sending mail "%s": %s', subject, body)

        message = EmailMessage()
        message['From'] = self._from
        message['To'] = self._to
        message['Subject'] = subject
        message.set_content(body)

        server = smtplib.SMTP(self._host, self._port)
        if self._starttls:
            server.starttls()
        if self._user is not None:
            server.login(self._user, self._password)
        server.send_message(message)
        server.quit()
