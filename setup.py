#!/usr/bin/env python
"""
	MMIT: Max Margin Interval Trees
	Copyright (C) 2017 Toby Dylan Hocking, Alexandre Drouin
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from platform import system as get_os_name

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Configure the compiler based on the OS
if get_os_name().lower() == "darwin":
    os_compile_flags = ["-mmacosx-version-min=10.9"]
else:
    os_compile_flags = []

# Required for the automatic installation of numpy
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

solver_module = Extension('mmit.core.solver',
                          language="c++",
                          sources=['mmit/core/solver_python_bindings.cpp',
                                   'mmit/core/solver.cpp',
                                   'mmit/core/piecewise_function.cpp',
                                   'mmit/core/coefficients.cpp'],
                          extra_compile_args=["-std=c++0x"] + os_compile_flags)

setup(
    name = "mmit",
    version = "2017.03.15",
    packages = find_packages(),

    cmdclass={'build_ext':build_ext},
    setup_requires = [],
    install_requires = [],

    author = "Toby Dylan Hocking, Alexandre Drouin",
    author_email = "toby.hocking@r-project.org, aldro61@gmail.com",
    maintainer="Alexandre Drouin",
    maintainer_email="aldro61@gmail.com",
    description="Max Margin Interval Trees",
    long_description = "Description: Fast O(P N log N) algorithm for learning a regression tree with interval censored "
                       "output data.",
    license = "GPL-3",
    keywords = "machine learning regression tree censored interval data",
    url = "https://github.com/aldro61/mmit",

    ext_modules = [solver_module],
    
    test_suite='nose.collector',
    tests_require=['nose']
)
