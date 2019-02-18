from torch.optim import Adam
from torch.utils.data import DataLoader

from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *


def show_normalized_image(img, ax=None, mono=False):
    if mono:
        img.numpy()[..., np.newaxis]
    np_img = img.numpy().transpose(1, 2, 0)
    lifted = np_img - np.min(np_img)
    ranged = lifted / np.max(lifted)
    show_np_image(ranged, ax=ax)


class MonoTo3ChLayer(nn.Module):
    def __init__(self):
        super(MonoTo3ChLayer, self).__init__()
    def forward(self, x):
        x.unsqueeze_(1)
        return x.repeat(1, 3, 1, 1)


def _get_model(weight_file, device, model_fn, mono):
    base_model = model_fn(pretrained=True)
    feature_model = nn.Sequential(*list(base_model.children())[:-1],
                                  nn.AdaptiveAvgPool2d(1),
                                  Flatten())
    # Load initial weights
    if weight_file is not None:
        feature_model.load_state_dict(torch.load(weight_file))
    # Add mono image input layer at the bottom of feature model
    if mono:
        feature_model = nn.Sequential(MonoTo3ChLayer(), feature_model)
    if device is not None:
        feature_model.to(device)

    feature_model.eval()
    return feature_model


def get_resnet101(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet101, mono=mono)


def get_resnet50(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet50, mono=mono)


def get_resnet34(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet34, mono=mono)


def get_resnet18(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet18, mono=mono)


def get_densenet121(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.densenet121, mono=mono)


def train_proto_net(args, model, device, n_epochs,
                    background_taskloader,
                    evaluation_taskloader,
                    path='.',
                    lr=3e-3,
                    drop_lr_every=100,
                    evaluation_episodes=100,
                    episodes_per_epoch=100,
                   ):
    print(f'Training Prototypical network...')

    # Prepare model
    model.to(device, dtype=torch.float)
    model.train(True)

    # Prepare training etc.
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.NLLLoss().cuda()
    ensure_folder(path + '/models')
    ensure_folder(path + '/logs')

    def lr_schedule(epoch, lr):
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr

    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=path + '/models/'+args.param_str+'_e{epoch:02d}.pth',
            monitor=args.checkpoint_monitor or f'val_{args.n_test}-shot_{args.k_test}-way_acc',
            period=args.checkpoint_period or 100,
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(path + f'/logs/{args.param_str}.csv'),
        background_taskloader.batch_sampler.callback,
    ]

    fit(
        model,
        optimizer,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        epoch_metrics=[f'val_{args.n_test}-shot_{args.k_test}-way_acc'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
    )
