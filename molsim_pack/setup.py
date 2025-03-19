import sys
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "molsim._molsim",
        sources=[
            "molsim/pybindings.cpp",
            "molsim/hardDisks/hardDisks.cpp",
            "molsim/molecularDynamics/md.cpp",
            "molsim/molecularDynamics/sample.cpp",
            "molsim/molecularDynamics/thermostats.cpp",
            "molsim/monteCarlo/mc.cpp",
            "molsim/monteCarlo/moves.cpp",
        ],
        include_dirs=[
            "molsim/hardDisks",
            "molsim/molecularDynamics",
            "molsim/monteCarlo",
            "molsim/utils",
        ],
        cxx_std=17,
    )
]

setup(
    name="molsim",
    version="1.0.0",
    description="Code accompanying exercises for the Molecular Simulation course.",
    author="van 't Hoff Institute of Molecular Sciences, University of Amsterdam",
    author_email="d.dubbeldam@uva.nl",
    python_requires=">=3.10",
    install_requires=[
        "pybind11",
        "numpy",
        "matplotlib",
        "scipy",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
        classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
)
