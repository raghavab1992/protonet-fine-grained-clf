from dlcliche.image import *
sys.path.append('..') # app
sys.path.append('../..') # root
from easydict import EasyDict
from app_utils_clf import *
from whale_plus_utils import *
from config import DATA_PATH
args = EasyDict()

#### LOOK HERE ####
name = 'k'
args.normalize = 'imagenet'
#### LOOK HERE ####

# Basic training parameters
args.distance = 'l2'
args.n_train = 1
args.n_test = 1
args.q_train = 1
args.q_test = 1

args.k_train = 60
args.k_test = 10
SZ = 224
RE_SZ = 256

args.n_epochs = 600
args.drop_lr_every = 30
args.lr = 3e-3
args.init_weight = None
args.part = 0
args.n_part = 1
args.augment_train = 'train'

data_train = DATA_PATH+'/train'
data_test  = DATA_PATH+'/test'
TRN_N_IMAGES = 2

args.param_str = f'{name}_k{args.k_train}'
args.checkpoint_monitor = 'categorical_accuracy'
args.checkpoint_period = 50

print(f'Training {args.param_str}.')

# Data
df = pd.read_csv(DATA_PATH+'/train.csv')
df = df[df.Id != 'new_whale']
ids = df.Id.values
classes = sorted(list(set(ids)))
images = df.Image.values
all_cls2imgs = {cls:images[ids == cls] for cls in classes}

trn_images = [image for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) >= TRN_N_IMAGES]
trn_labels = [_id   for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) >= TRN_N_IMAGES]
val_images = [image for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) == 2]
val_labels = [_id   for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) == 2]

args.episodes_per_epoch = len(trn_images) // args.k_train + 1
args.evaluation_episodes = 50 # setting small value, anyway validation set is almost useless here

print(f'Samples = {len(trn_images)}, {len(val_images)}')

# Model
feature_model = get_resnet50(device=device, weight_file=args.init_weight)

# Dataloader
background = WhaleImagesPlus(data_train, trn_images, trn_labels, re_size=RE_SZ, to_size=SZ, augment=args.augment_train,
                             part=args.part, n_part=args.n_part, normalize=args.normalize)
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=8
)
evaluation = WhaleImagesPlus(data_train, val_images, val_labels, re_size=RE_SZ, to_size=SZ, augment='test',
                             part=args.part, n_part=args.n_part, normalize=args.normalize)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=8
)

# Train
train_proto_net(args,
                model=feature_model,
                device=device,
                path=name,
                n_epochs=args.n_epochs,
                background_taskloader=background_taskloader,
                evaluation_taskloader=evaluation_taskloader,
                drop_lr_every=args.drop_lr_every,
                evaluation_episodes=args.evaluation_episodes,
                episodes_per_epoch=args.episodes_per_epoch,
                lr=args.lr,
               )
torch.save(feature_model.state_dict(), f'{name}/{args.param_str}_epoch{args.n_epochs}.pth')
