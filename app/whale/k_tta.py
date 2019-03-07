from dlcliche.image import *
sys.path.append('..') # app
sys.path.append('../..') # root
from easydict import EasyDict
from app_utils_clf import *
from whale_plus_utils import *
from config import DATA_PATH

calculate_results_plus(weight_file='k/k_k60_epoch600.pth',
                       output_path='results',
                       SZ=224,
                       get_model_fn=get_resnet50,
                       device=device,
                       train_csv=DATA_PATH+'/train.csv',
                       data_train='images/train-448-AC-CR',
                       data_test='images/test-448-AC-CR',
                       data_type='normal',
                       normalize='imagenet',
                       N_TTA=4)

