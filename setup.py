# -*- coding: utf-8 -*-
import subprocess
from distutils.command.build import build as _build

from setuptools import setup, find_packages, Command


class build(_build):  # pylint: disable=invalid-name
    sub_commands = _build.sub_commands + [('CustomCommands', None)]


CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', 'install', '-y', '--no-install-recommends',
     'build-essential', 'python-tk'],
    ['apt-get', 'clean'],
    ['rm', '-rf', '/var/lib/apt/lists/*'],
    ['pip', 'install', '-U', 'pip'],
    ['pip', 'install', 'google-cloud', 'h5py', 'matplotlib'],
]


class CustomCommands(Command):

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print('Running command: %s' % command_list)
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        print('Command output: %s' % stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)


REQUIRED_PACKAGES = [
]

setup(
    name='imdb-sympathizers',
    description='imdb review classification by Cloud ML',
    author='chck',
    author_email='deadline.is.today@gmail.com',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    cmdclass={'build': build, 'CustomCommands': CustomCommands})
