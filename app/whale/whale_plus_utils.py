from whale_utils import *
from PIL import Image #import PIL


def partition_np_image(image, part, n_part=2):
    ih, iw = image.shape[:2]
    assert part < n_part
    M = 2
    dw = iw // (n_part * M + 1) # 1/5 if n_part=2
    x0 = dw * part * M
    x1 = min(x0 + (M + 1) * dw, iw)
    image = image[:, x0:x1, :]
    return image


def get_aug_plus(re_size=224, to_size=224, augment='train', normalize='imagenet'):
    augs = [A.Resize(height=re_size, width=re_size)]
    if augment == 'train':
        augs.extend([
            A.RandomCrop(height=to_size, width=to_size),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=30, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.75),
            A.Blur(p=0.5),
            A.Cutout(max_h_size=to_size//12, max_w_size=to_size//12, p=0.5),
        ])
    elif augment == 'train_hard':
        augs.extend([
            A.RandomCrop(height=to_size, width=to_size),
            A.IAAAffine(scale=1.3, translate_percent=0.2, translate_px=None,
                                        rotate=40, shear=20),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.75),
            A.IAAPerspective(p=1),
            A.IAAAdditiveGaussianNoise(p=0.2),
            A.Blur(p=0.5),
            A.Cutout(max_h_size=to_size//12, max_w_size=to_size//12, p=0.5),
        ])
    elif augment == 'test':
        augs.extend([
            A.CenterCrop(height=to_size, width=to_size)
        ])
    elif augment == 'tta':
        augs.extend([
            A.IAAAffine(scale=1.05, translate_percent=0.1, translate_px=None, 
                        rotate=20, shear=10),
            A.CenterCrop(height=to_size, width=to_size),
        ])
    else:
        raise Exception(f'aug level not supported: {augment}')
    return A.Compose(augs + [A.Normalize(samplewise=(normalize=='samplewise'))])


class WhaleImagesPlus(WhaleImages):
    def __init__(self, path, images, labels, re_size=256, to_size=224, part=0, n_part=1,
                 augment='normal', normalize='samplewise'):
        super().__init__(path, images, labels, re_size=re_size, to_size=to_size, train=(augment=='train'))
        self.transform = get_aug_plus(re_size=re_size, to_size=to_size, augment=augment, normalize=normalize)
        self.part, self.n_part = part, n_part

    def __getitem__(self, item):
        instance = self.loader(self.datasetid_to_filepath[item])
        if self.n_part > 1:
            instance = partition_np_image(instance, self.part, n_part=self.n_part)
        instance = self.transform(image=instance)['image']
        instance = self.to_tensor(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label


def calculate_results_plus(weight_file, output_path, SZ, get_model_fn, device, train_csv='data/data.csv',
                           data_train='data/train', data_test='data/test',
                           data_type='normal', normalize='samplewise', part=0, n_part=1, N_TTA=4):
    weight_file = Path(weight_file)
    output_path = Path(output_path)
    ensure_folder(output_path)
    submission_file_stem = ('NS_' if normalize=='samplewise' else '') + weight_file.stem

    # Training samples
    df = pd.read_csv(train_csv)
    df = df[df.Id != 'new_whale']
    images = df.Image.values
    labels = df.Id.values

    # Test samples
    test_images = get_test_images(data_test)
    dummy_test_gts = list(range(len(test_images)))

    print(f'Training samples: {len(images)}, # of labels: {len(list(set(labels)))}.')
    print(f'Test samples: {len(test_images)}.')
    print(f'Work in progress for {submission_file_stem}...')

    # Making dataloaders
    def get_dl(images, labels, folder, SZ=SZ, batch_size=64, augment='test', normalize='samplewise'):
        if data_type == 'normal':
            ds = WhaleImagesPlus(folder, images, labels, re_size=SZ, to_size=SZ,
                                 augment=augment, normalize=normalize, part=part, n_part=n_part)
        else:
            raise ValueError('invalid data type')
        dl = DataLoader(ds, batch_size=batch_size)
        return dl

    # 1. NORMAL RESULT
    # Make prototypes
    trn_dl = get_dl(images, labels, data_train)
    model = get_model_fn(device=device, weight_file=weight_file)
    proto_net = ExtModelProtoNetClf(model, trn_dl.dataset.classes, device)

    proto_net.make_prototypes(trn_dl)

    # Calculate distances
    test_dl = get_dl(test_images, dummy_test_gts, data_test)
    test_embs, gts = proto_net.get_embeddings(test_dl)
    test_dists = proto_net.predict_embeddings(test_embs, softmax=False)

    np.save(output_path/f'test_dists_{submission_file_stem}.npy', test_dists)
    np.save(output_path/f'prototypes_{submission_file_stem}.npy', np.array([x.mean() for x in proto_net.prototypes]))

    # 2. PTA RESULT
    print(f'Work in progress for PTA_{submission_file_stem}...')
    trn_dl = get_dl(images, labels, data_train, augment='tta', normalize=normalize)
    proto_net.make_prototypes(trn_dl, repeat=N_TTA, update=True)

    test_dists = proto_net.predict_embeddings(test_embs, softmax=False)

    np.save(output_path/f'test_dists_PTA_{submission_file_stem}.npy', test_dists)
    np.save(output_path/f'prototypes_PTA_{submission_file_stem}.npy', np.array([x.mean() for x in proto_net.prototypes]))

    # 3. PTTA RESULT
    print(f'Work in progress for PTTA_{submission_file_stem}...')
    test_dl = get_dl(test_images, dummy_test_gts, data_test, augment='tta', normalize=normalize)
    tta_embs = []
    for i in range(N_TTA):
        embs, gts = proto_net.get_embeddings(test_dl)
        tta_embs.append(embs)
    all_test_embs = np.array([test_embs] + tta_embs)
    mean_test_embs = all_test_embs.mean(axis=0)

    test_dists = proto_net.predict_embeddings(mean_test_embs, softmax=False)

    np.save(output_path/f'test_dists_PTTA_{submission_file_stem}.npy', test_dists)
