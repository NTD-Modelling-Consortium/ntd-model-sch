import setuptools

setuptools.setup(
    name='sch_simulation',
    version='2.0.1dev',
    url='https://www.ntdmodelling.org',
    maintainer='ArtRabbit',
    maintainer_email='support@artrabbit.com',
    description='SCH simulation model',
    long_description='Individual-based model in Medley 1989 thesis and Anderson&Medley 1985.',
    packages=setuptools.find_packages(),
    python_requires='>=3.9,<3.11',
    install_requires=['numpy', 'scipy', 'pandas', 'joblib', 'matplotlib', 'openpyxl', 'pytest'],
    include_package_data=True
)
