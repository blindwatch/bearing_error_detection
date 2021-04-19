from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.TRAIN = edict()

__C.DATASET = './prodata/'
__C.WEIGHTS = './cache/weight.pth'
__C.WEIGHTS_BEST = './cache/weight_best.pth'

__C.FIRST_EPOCH = './first_epoch.csv'
__C.LOSS_PLOT = './loss_plot.csv'
__C.ACC_PLOT = './acc_plot.csv'
__C.CATEGORY_ACC_TRAIN = './category_acc_train.csv'
__C.CATEGORY_ACC_VAL = './category_acc_val.csv'
__C.STATE = 'Test'

__C.RESULT_PIC = './result/'

__C.NET_SET = [26, 128, 64, 10]

__C.TRAIN.FROM_CHECK= False
__C.TRAIN.NUM_ITERATION = 50
__C.TRAIN.BATCH_SIZE = 600
__C.TRAIN.SHUFFLE = True
__C.TRAIN.DEFAULT_LEARNING_RATE = 1e-3
__C.TRAIN.MIXVAL = True
__C.TRAIN.MIXEPOCH = {"epoch": 30}
__C.TRAIN.SAVE_FREQ = 5
__C.TRAIN.PRINT_FREQ = 10
