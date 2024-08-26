from setuptools import setup, find_packages

setup(
    name='ai-researcher',
    author='Chenglei Si',
    author_email='clsi@stanford.edu',
    description='Official implementation of the Research Ideation Agent by Stanford NLP',
    url='https://github.com/NoviScl/AI-Researcher',
    package_dir={'': 'src'},
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'setuptools',
    ]
)