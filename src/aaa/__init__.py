import os
import aaa.data

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DATA_DIR = os.path.dirname(aaa.data.__file__)
_DEMO_DATA_DIR = os.path.join(_ROOT_DIR, 'demo_data')
