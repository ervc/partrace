from setuptools import setup
import sys

sys.path.insert(0, "partrace")
import partrace as pt

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()


setup(
    name='partrace',
    version=pt.__version__,
    description='Particle tracking wrapper for use with FARGO3D outputs',
    url='https://github.com/ervc/partrace',
    author=pt.__author__,
    author_email='ericvc@uchicago.edu',
    packages=['partrace'],
    install_requires = install_requires,
)