import sysconfig
import sys


paths = sysconfig.get_paths()

# 'C:\\Users\\Xiaomi\\AppData\\Local\\Programs\\Python\\Python38-32\\include'
# 'C:\\Users\\Xiaomi\\AppData\\Local\\Programs\\Python\\Python38-32\Lib\\site-packages\\numpy\\core\\include\\numpy'
def get_platform_postfix():
    if sys.platform == 'win32':
        return '\\numpy\\core\\include\\numpy'
    else:
        return '/numpy/core/include/numpy'

PYTHON_INCLUDE = paths['include']
NUMPY_INCLUDE = paths['purelib'] + get_platform_postfix()

