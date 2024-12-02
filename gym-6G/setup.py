# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from setuptools import setup

setup(
    name="gym-6G",
    version="1.0",
    install_requires=[
        "gymnasium==0.28.1",
        "matplotlib==3.9.0",
        "scipy==1.13.1"]
)
