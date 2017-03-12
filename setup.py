#!/usr/bin/env python

"""
MMIT
"""
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

solver_module = Extension('mmit/core/solver', sources=['mmit/core/solver_python_bindings.cpp',
                                                       'mmit/core/solver.cpp'])

setup(
    name = "mmit",
    version = "0.0.1",
    packages = find_packages(),

    cmdclass={'build_ext':build_ext},
    setup_requires = [],
    install_requires = [],

    author = "",
    author_email = "",
    description = "",
    license = "GPLv3",
    keywords = "",
    url = "",

    ext_modules = [solver_module]
)
