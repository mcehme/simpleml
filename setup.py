from setuptools import setup, find_packages

setup(
    name='simpleml',
    version='0.0.1',
    description = 'A simple ML library',
    author = 'Michael Ehme',
    packages=find_packages(),
    install_requires=['numpy']
)
