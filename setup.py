from setuptools import find_packages, setup 

with open('README.md', 'r') as f:
	long_description = f.read()

setup(
	name='shap_selection',
	packages=find_packages(include=['shap_selection']),
	version='0.1.6',
	description='Selecting features using SHAP values',
	long_description=long_description,
    long_description_content_type='text/markdown',
	author='Wilson Estecio Marcilio Junior',
	author_email='wilson_jr@outlook.com',
	url='https://github.com/wilsonjr/SHAP_FSelection',
	license='MIT',
	install_requires=['shap', 'numpy'],
	setup_requires=['pytest-runner'],
	tests_require=['pytest'],
	test_suite='tests',
)