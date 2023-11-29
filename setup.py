from setuptools import setup, find_packages

from spectrumlab import DESCRIPTION, VERSION, NAME, AUTHOR_NAME, AUTHOR_EMAIL


setup(
	# info
    name=NAME,
	description=DESCRIPTION,
	license='MIT',
    keywords=['spectroscopy', 'spectra emulation', 'spectra process'],

	# version
    version=VERSION,

	# author details
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,

	# setup directories
    packages=find_packages(),

	# setup data
    include_package_data=True,

	# requires
    install_requires=['tqdm', 'numpy', 'pandas', 'matplotlib', 'scipy'],
    python_requires='>=3.10',

)
