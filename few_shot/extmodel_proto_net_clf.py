"""
For testing what if we use ImageNet pretrained model as ProtoNet??
"""
from dlcliche.utils import *
from dlcliche.math import *
from dlcliche.image import show_np_image, subplot_matrix

from torchvision import models
from torch import nn
import torch
from tqdm import tqdm

# TODO: Support cpu environment

class BasePretrainedModel(nn.Module):
    def __init__(self, base_model=models.resnet18, n_embs=512, print_shape=False):
        super(BasePretrainedModel, self).__init__()
        resnet = base_model(pretrained=True)
        self.body = nn.Sequential(*list(resnet.children())[:-1])
        self.n_embs = n_embs
        self.print_shape = print_shape

    def forward(self, x):
        x = self.body(x)
        if self.print_shape:
            print(x.shape)
        return x.view(-1, self.n_embs)


class ExtModelProtoNetClf(object):
    """ProtoNet as conventional classifier using external model.
    Created for testing what if we use ImageNet pretrained model for getting embeddings.

    TODO Fix bad design for member-call-order dependency...
    """

    def __init__(self, model, classes, device, n_embeddings=None):
        model.to(device)
        model.eval()
        self.model = model
        self.classes = classes
        self.device = device
        self.n_embeddings = n_embeddings # First get_embeddings() will set this, if it is None
        self.n_classes = len(classes)
        self._make_null_prototypes()
        self.log = get_logger()

    def _make_null_prototypes(self):
        if self.n_embeddings:
            self.prototypes = [OnlineStats(self.n_embeddings) \
                               for _ in range(self.n_classes)]
        else:
            self.prototypes = None

    def get_embeddings(self, dl, visualize=False):
        """Get embeddings for all samples available in dataloader."""
        gts, cur = [], 0
        with torch.no_grad():
            for batch_index, (X, y_gt) in tqdm(enumerate(dl), total=len(dl)):
                dev_X, y_gt = X.to(self.device), list(y_gt)
                this_embs = self.model(dev_X).cpu().detach().numpy()
                if cur == 0:
                    self.n_embeddings = this_embs.shape[-1]
                    embs = np.zeros((len(dl.dataset), self.n_embeddings))

                if visualize:
                    for i, ax in enumerate(subplot_matrix(columns=4, rows=2, figsize=(16, 8))):
                        if len(dl) <= batch_index * 8 + i: break
                        show_np_image(np.transpose(X[i].cpu().detach().numpy(), [1, 2, 0]), ax=ax)
                    plt.show()

                for i in range(len(this_embs)):
                    embs[cur] = this_embs[i]
                    gts.append(y_gt[i])
                    cur += 1
        return np.array(embs), gts

    def make_prototypes(self, support_set_dl, repeat=1, update=False, visualize=False):
        """Calculate prototypes by accumulating embeddings of all samples in given support set.
        Args:
             support_set_dl: support set dataloader.
             repeat: test parameter for what if we get prototype with augmented samples.
             update: set True if you don't want to update prototypes with new samples from dataloader.
        """
        # Get embeddings of support set samples
        embs, gts = self.get_embeddings(support_set_dl, visualize=visualize)
        # Make prototypes if not there
        if update:
            self.log.info('Using current prototypes.')
        else:
            self.log.info('Making new prototypes.')
            self._make_null_prototypes()
        # Update prototypes (just by feeding to online stat class)
        for i in range(repeat):
            for emb, cls in zip(embs, gts):
                if not isinstance(cls, int):
                    cls = self.classes.index(cls)
                self.prototypes[cls].put(emb)
            if i < repeat - 1:
                embs, gts = self.get_embeddings(support_set_dl)  # no visualization

    def get_prototypes(self):
        return np.array([p.mean() for p in self.prototypes])

    def save_prototypes(self, filename):
        np.save(filename, self.get_prototypes())

    def load_prototypes(self, filename):
        prototypes = np.load(filename)
        for i, prototype in enumerate(prototypes):
            self.prototypes[i].set_mean(prototype)

    def predict_embeddings(self, X_embs, softmax=True, normalized_softmax=True):
        preds = np.zeros((len(X_embs), self.n_classes))
        proto_embs = [p.mean() for p in self.prototypes]
        for idx_sample, x in tqdm(enumerate(X_embs), total=len(X_embs)):
            for idx_class, proto in enumerate(proto_embs):
                preds[idx_sample, idx_class] = -(np.linalg.norm(x - proto)**2)
        if softmax:
            if normalized_softmax:
                preds /= np.max([1.0, np.abs(preds.mean())])
            preds = np_softmax(preds)
        return preds

    def predict(self, data_loader):
        embs, y_gts = self.get_embeddings(data_loader)
        return self.predict_embeddings(embs), y_gts

    def evaluate(self, data_loader):
        y_hat, y_gts = self.predict(data_loader)
        return calculate_clf_metrics(y_gts, y_hat)

    @staticmethod
    def get_uncertainty(dists):
        _max = np.max(dists, axis=-1)
        _mean = np.mean(dists, axis=-1)
        uncertainty = _max/_mean
        return uncertainty
