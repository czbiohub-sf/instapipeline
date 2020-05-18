from setuptools import setup, find_packages
import os

install_requires = [
    line.rstrip() for line in open(
        os.path.join(os.path.dirname(__file__), "requirements.txt")
    )
]

setup(
	name='fishanno',
	install_requires=install_requires,
	version='0.1',
	package_dir={'fishanno':'fishanno'},
	packages=find_packages(),
	license='MIT',
	long_description=open('README.md').read(),
	author='jenny.vophamhi'
)