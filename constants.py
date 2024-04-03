import os
import sys

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
DIM = 3
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = f'{PROJECT_ROOT}/assets/'
RAW_ROOT = f'{DATA_ROOT}raw/'
CHECKPOINTS_ROOT = f'{DATA_ROOT}checkpoints/'

mnt = DATASET_PATH
# CACHE_ROOT = '/mnt/amir/cache/'
CACHE_ROOT = f'{mnt[:-1]}/ShapeNetCore_cache/'

UI_OUT = f'{DATA_ROOT}ui_export/'
POINTS_CACHE = f'{DATA_ROOT}points_cache/'
DualSdfData =  f'{RAW_ROOT}dualSDF/'
UI_RESOURCES = f'{DATA_ROOT}/ui_resources/'


# mnt = '/mnt/amir' if os.path.isdir('/mnt/amir') else '/data/amir'
# Shapenet_WT = f'{mnt}/ShapeNetCore_wt/'
# Shapenet = f'{mnt}/ShapeNetCore.v2/'
# MANIFOLD_SCRIPT = "/home/amir/projects/Manifold/build"
mnt = DATASET_PATH
Shapenet_WT = f'{mnt[:-1]}/ShapeNetCore_wt' #TODO: cannot download ShapeNet_wt
Shapenet = f'{mnt[:-1]}/ShapeNetCore_v2/'
MANIFOLD_SCRIPT = "/Users/liujunyu/Desktop/Research/BVC/ITSS/code/Manifold/build"

COLORS = [[231, 231, 91], [103, 157, 200], [177, 116, 76], [88, 164, 149],
         [236, 150, 130], [80, 176, 70], [108, 136, 66], [78, 78, 75],
         [41, 44, 104], [217, 49, 50], [87, 40, 96], [85, 109, 115], [234, 234, 230],
          [30, 30, 30]]

GLOBAL_SCALE = 10
MAX_GAUSIANS = 32
MAX_VS = 100000

