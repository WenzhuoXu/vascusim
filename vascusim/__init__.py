import os
import sys
import numpy as np
import ctypes
import platform


from pkg_resources import get_distribution, DistributionNotFound


try:
    # The package is installed
    __version__ = get_distribution("vascusim").version
except DistributionNotFound:
    # The package is not installed, so we can set a default version
    __version__ = "0.1.0"
# Set the path to the directory where this file is located

home_dir = os.path.expanduser('~')
package_dir = os.path.dirname(os.path.abspath(__file__))


