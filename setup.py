from setuptools import setup
import partrace as pt

setup(
    name='partrace',
    version=pt.__version__,
    description='Particle tracking wrapper for use with FARGO3D outputs',
    author=pt.__author__,
    packages=['partrace'])