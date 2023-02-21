from setuptools import setup
setup(
    name='Synthetic_ PV_Profiles',
    version='1.0',
    author='Mahrokh Ghoddousiboroujeni',
    description='Module for simulating synthetic PV power generation profiles based on PV installation setups and given weather and location information.',
    long_description='Can be used to benchmark federated learning algorithms',
    url='https://github.com/MahrokhGB/Synthetic_PV_Profiles',
    keywords='PV, population models, simulated data',
    python_requires='==3.9.9',
    install_requires=[
        'matplotlib==3.5.3',
        'numpy==1.17.4',
        'pandas==1.4.4',
        'pvlib==0.9.3',
        'scikit_learn==1.2.1',
        'scipy==1.9.1',
        'statsmodels==0.13.5',
        'jupyter'
    ]
)

# TODO: add package_data, entry_points, packages=find_packages(include=['exampleproject', 'exampleproject.*']),