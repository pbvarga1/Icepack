from setuptools import setup


setup(
    name='icepack',
    version='0.1.0',
    description='Package for running icepack scripts using python',
    author='Perry Vargas',
    author_email='perrybvargas@gmail.com',
    packages=[
        'icepack',
    ],
    package_dir={'icepack':
                 'icepack'},
    include_package_data=True,
    install_requires=[
        'Jinja2==2.10',
        'netCDF4==1.4.3.2',
        'numpy==1.16.2',
        'pandas==0.24.2',
        'python-dateutil==2.8.0',
        'scipy==1.2.1',
        'h5py==2.9.0',
    ],
    license="BSD",
    zip_safe=False,
    keywords='web',
    entry_points={
        "console_scripts": [
            "run_icepack = icepack.run_icepack:cli",
            "create_pump_data = icepack.create_pump_data:cli",
            "run_case_inputs = icepack.run_icepack:cli2",
        ]
    },

)
