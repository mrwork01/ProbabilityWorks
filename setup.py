from setuptools import setup, find_packages

setup(
	name='ProbabilityWorks',
	version='0.1.0',
	description='Statistical Library',
	author='Marc Work',
	license='MIT',
	packages=['ProbabilityWorks'],
	include_package_data=True,
	install_requires=['pandas']

	)