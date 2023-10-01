from setuptools import find_packages, setup

setup(
    name="complexPyTorch",
    version="0.4.1",
    description="A high-level toolbox for using complex valued neural networks in PyTorch.",
    long_description=open("README.md").read().strip(),
    long_description_content_type="text/markdown",
    author="Sebastien M. Popoff",
    author_email="sebastien.popoff@espci.psl.eu",
    url="https://gitlab.institut-langevin.espci.fr/spopoff/complexPyTorch",
    packages=find_packages(),
    install_requires=["torch"],
    python_requires=">=3.6",
    license="MIT License",
    zip_safe=False,
    keywords="pytorch, deep learning, complex values",
    classifiers=[""],
)
