from dlcliche.image import *
from dlcliche.math import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
from IPython.display import display

sys.path.append('..') # app
sys.path.append('../..') # root
from utils import top_5_pred_labels
from app_utils_clf import *

def _get_test_images(data_test):
    return [str(f).replace(data_test+'/', '') for f in Path(data_test).glob('*.jpg')]


def plot_prototype_2d_space_distribution(prototypes):
    X = prototypes
    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    print('PCA: Explained variance ratio: %s'
          % str(pca.explained_variance_ratio_))
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=.6)
    plt.title('Prototype Distribution PCA')
    plt.xlim((-4, 4))
    plt.ylim((-3, 3))
    plt.show()
    return X_pca


def plot_prototype_3d_space_distribution(prototypes):
    X = prototypes
    pca = PCA(n_components=3)
    X_pca = pca.fit(X).transform(X)
    print('PCA: Explained variance ratio: %s'
          % str(pca.explained_variance_ratio_))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X_pca[:, 0],X_pca[:, 1],X_pca[:, 2])
    ax.set_title('Prototype Distribution PCA')
    ax.set_xlim((-4, 4))
    ax.set_ylim((-3, 3))
    ax.set_zlim((-3, 3))
    plt.show()
    return X_pca


def get_classes(data='data', except_new_whale=True, append_new_whale_last=True):
    df = pd.read_csv(data+'/train.csv')
    if except_new_whale:
        df = df[df.Id != 'new_whale']
    classes = sorted(list(set(df.Id.values)))
    if append_new_whale_last:
        classes.append('new_whale')
    return classes


def prepare_submission(submission_filename, test_dists, new_whale_thresh, data_test, classes):
    def _create_proto_submission(preds, name, classes):
        sub = pd.DataFrame({'Image': _get_test_images(data_test)})
        sub['Id'] = [classes[i] if not isinstance(i, str) else i for i in 
                     top_5_pred_labels(torch.tensor(preds), classes)]
        ensure_folder('subs')
        sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')

    dist_new_whale = np.ones_like(test_dists[:, :1])
    dist_new_whale[:] = new_whale_thresh
    final_answer = np.c_[test_dists, dist_new_whale]

    _create_proto_submission(final_answer, submission_filename, classes)
    print(submission_filename,
          pd.read_csv(f'subs/{submission_filename}.csv.gz').Id.str.split().apply(lambda x: x[0] == 'new_whale').mean(),
          len(set(pd.read_csv(f'subs/{submission_filename}.csv.gz').Id.str.split().apply(lambda x: x[0]).values)))
    display(pd.read_csv(f'subs/{submission_filename}.csv.gz').head())
