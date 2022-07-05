from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sklearnBPMF',
    version='0.0.4',
    author='Ola Oni',
    author_email='oa.oni7@gmail.com',
    description='Sklearn wrapper for Bayesian Probabilistic Matrix Completion with Macau, Smurff, and Bayesian Regression',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/oaoni/sklearnBPMF',
    project_urls = {
        "Bug Tracker": "https://github.com/oaoni/sklearnBPMF/issues"
    },
    license='MIT',
    packages=find_packages(include=['sklearnBPMF','sklearnBPMF.*']),
    install_requires=['smurff','umap-learn','scipy','networkx','sklearn','pandas','numpy'],
)
