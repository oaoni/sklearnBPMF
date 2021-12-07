import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sklearnBPMF',
    version='0.0.1',
    author='Ola Oni',
    author_email='oa.oni7@gmail.com',
    description='Sklearn wrapper for Bayesian Probabilistic Matrix Completion with Macau and Smurff',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/oaoni/sklearnBPMF',
    project_urls = {
        "Bug Tracker": "https://github.com/oaoni/sklearnBPMF/issues"
    },
    license='MIT',
    packages=['sklearnBPMF','sklearnBPMF.*'],
    install_requires=['smurff','umap','scipy','networkx','sklearn','pandas','numpy'],
)
