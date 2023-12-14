import os
import sys 
import logging
import numpy as np

########################## SET RANDOM SEEDS #################################
np.random.seed(2)

############################# LIBRARY IMPORTS ####################################################
assert __file__[-1] != '/' , f'File:{__init__}, cannot be parsed' 
SRC_DIR,_ = os.path.split(os.path.abspath(__file__))
HOME_DIR,_ = os.path.split(SRC_DIR)
sys.path.extend([HOME_DIR])

# ############################ FOLDER PATHS #######################################################
# Folders
DATASET_DIR = os.path.join(HOME_DIR,'dataset') # Path containing all the training data (currently using xyz)
MESH_PATH = os.path.join(DATASET_DIR,'models','models')
RENDER_DIR = os.path.join(HOME_DIR,'rendered_videos')
LOG_DIR = os.path.join(HOME_DIR,'logs')

## Dataset split paths 
TRAIN_PATH = os.path.join(DATASET_DIR,"training_data/training_data_filtered/training_data/")
TEST_PATH = os.path.join(DATASET_DIR,"testing_data_pose/testing_data_pose_filtered/testing_data/")

# ############################ DATASET CONSTANTS #######################################################
# Excercise categories 
NUM_OBJECTS=79
VALID_CLASS_INDICES = [1, 5, 8, 13, 14, 18, 20, 21, 26, 29, 30, 35, 39, 42, 43, 48, 50, 51, 52, 55, 56, 57, 58] 
MESHID2CLASS = dict([ (x,i)  for  i,x in enumerate(VALID_CLASS_INDICES)])  
LABELS = ['', 'a_lego_duplo', '', '', '', 'b_lego_duplo', '', '', 'bleach_cleanser', '', '', '', '', 'c_toy_airplane', 'cracker_box', '', '', '', 'd_toy_airplane', '', 'e_lego_duplo', 'e_toy_airplane', '', '', '', '', 'foam_brick', '', '', 'g_lego_duplo', 'gelatin_box', '', '', '', '', 'jenga', '', '', '', 'master_chef_can', '', '', 'mustard_bottle', 'nine_hole_peg_test', '', '', '', '', 'potted_meat_can', '', 'prism', 'pudding_box', 'rubiks_cube', '', '', 'sugar_box', 'tomato_soup_can', 'tuna_fish_can', 'wood_block', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
MAX_OBJECTS_IN_SAMPLE = 10

############################# POSE Estimation HYPERPARAMETERS #######################################################
CUDA=True
TRAIN_BATCH_SIZE=16
RENDER=True

############################# LOGGING #######################################################
DEBUG = True
class CustomFormatter(logging.Formatter):

	BLACK = '\033[0;30m'
	RED = '\033[0;31m'
	GREEN = '\033[0;32m'
	BROWN = '\033[0;33m'
	BLUE = '\033[0;34m'
	PURPLE = '\033[0;35m'
	CYAN = '\033[0;36m'
	GREY = '\033[0;37m'

	DARK_GREY = '\033[1;30m'
	LIGHT_RED = '\033[1;31m'
	LIGHT_GREEN = '\033[1;32m'
	YELLOW = '\033[1;33m'
	LIGHT_BLUE = '\033[1;34m'
	LIGHT_PURPLE = '\033[1;35m'
	LIGHT_CYAN = '\033[1;36m'
	WHITE = '\033[1;37m'

	RESET = "\033[0m"

	format = "[%(filename)s:%(lineno)d]: %(message)s (%(asctime)s) "

	FORMATS = {
		logging.DEBUG: YELLOW + format + RESET,
		logging.INFO: GREY + format + RESET,
		logging.WARNING: LIGHT_RED + format + RESET,
		logging.ERROR: RED + format + RESET,
		logging.CRITICAL: RED + format + RESET
	}

	def format(self, record):
		log_fmt = self.FORMATS.get(record.levelno)
		formatter = logging.Formatter(log_fmt)
		return formatter.format(record)

def get_logger(task_name=None):
	global LOG_DIR
	if task_name is None: 
		task_name = 'tmp'

	LOG_DIR = os.path.join(LOG_DIR, task_name)

	os.makedirs(LOG_DIR,exist_ok=True)

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)

	handler = logging.FileHandler(os.path.join(LOG_DIR,"log.txt"))
	handler.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)
	formatter = logging.Formatter(
		'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	handler = logging.StreamHandler()
	handler.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)
	handler.setFormatter(CustomFormatter())
	logger.addHandler(handler)

	try: 
		from tensorboardX import SummaryWriter
		writer = SummaryWriter(LOG_DIR)

	except ModuleNotFoundError:
		logger.warning("Unable to load tensorboardX to write summary.")
		writer = None

	return logger, writer

# logger,writer = get_logger() // Define logger as global variable in case of immediate debugging requirements.