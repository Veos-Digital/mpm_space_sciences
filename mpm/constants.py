import os
import mpm

STATIC_FOLDER = os.path.join(os.path.dirname(os.path.dirname(mpm.__file__)), 
                             "static")
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(mpm.__file__)), 
                             "data")