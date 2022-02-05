from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="scheptk",
    version="0.0.3",
    author="Jose M Framinan",
    author_email="framinan@us.es",
    description="Python scheduling package",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/framinan/scheptk",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)