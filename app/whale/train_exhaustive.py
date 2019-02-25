from dlcliche.image import *
sys.path.append('..') # app
sys.path.append('../..') # root
from easydict import EasyDict
from app_utils_clf import *
from whale_utils import *
from config import DATA_PATH

# Basic training parameters
args = EasyDict()
args.distance = 'l2'
args.n_train = 1
args.n_test = 1
args.q_train = 1
args.q_test = 1

args.k_train = 50
args.k_test = 10
SZ = 224
RE_SZ = 256

args.n_epochs = 100
args.drop_lr_every = 50
args.lr = 3e-3
args.init_weight = None

data_train = DATA_PATH+'/train'
data_test  = DATA_PATH+'/test'

args.param_str = f'app_whale_n{args.n_train}_k{args.k_train}_q{args.q_train}'
args.checkpoint_monitor = 'categorical_accuracy'
args.checkpoint_period = 50

print(f'Training {args.param_str}.')

# Data - 'more_than_two' or 'exhaustive'
trn_images, trn_labels, val_images, val_labels = get_training_data_lists('exhaustive')

args.episodes_per_epoch = len(trn_images) // args.k_train + 1
args.evaluation_episodes = 100 # setting small value, anyway validation set is almost useless here

print(f'Samples = {len(trn_images)}, {len(val_images)}')

# Model
feature_model = get_resnet18(device=device, weight_file=args.init_weight)

# Dataloader
background = WhaleImages(data_train, trn_images, trn_labels, re_size=RE_SZ, to_size=SZ)
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=8
)
evaluation = WhaleImages(data_train, val_images, val_labels, re_size=RE_SZ, to_size=SZ, train=False)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=8
)

# Train
train_proto_net(args,
                model=feature_model,
                device=device,
                path='.',
                n_epochs=args.n_epochs,
                background_taskloader=background_taskloader,
                evaluation_taskloader=evaluation_taskloader,
                drop_lr_every=args.drop_lr_every,
                evaluation_episodes=args.evaluation_episodes,
                episodes_per_epoch=args.episodes_per_epoch,
                lr=args.lr,
               )
torch.save(feature_model.state_dict(), f'{args.param_str}_epoch{args.n_epochs}.pth')