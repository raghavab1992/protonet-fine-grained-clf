# (Using) Prototypical Networks as a Fine Grained Classifier

This repository is heavily based on [Oscar Knagg](https://towardsdatascience.com/@oknagg)'s few-shot learning
implementation [github.com/oscarknagg/few-shot](https://github.com/oscarknagg/few-shot), focused on applying simple but strong [Prototpyical Networks](https://arxiv.org/pdf/1703.05175.pdf) to fine grained classification task.

Unlike very clean original implementation, this repository contains
some dirty code to quickly present sample solution to a Kaggle competition
"[Humpback Whale Identification](https://www.kaggle.com/c/humpback-whale-identification/)".

Some of competition submission code borrows functions from [Radek Osmulski](https://medium.com/@radekosmulski)'s github repository. Thank you.

I'd like to express sincere appreciation to both [Oscar Knagg](https://towardsdatascience.com/@oknagg) and [Radek Osmulski](https://medium.com/@radekosmulski).

![fig](assets/proto_nets_diagram.png)

Figure from original paper. Color circles: training samples, $c_i$: prototypes, $x$: test sample.

## Quick start

This project derives prerequisite below:

    This project is written in python 3.6 and Pytorch and assumes you have
    a GPU.

1. Install dl-cliche, excuse me this is my almost-private library to repeat cliche code.

    pip install git+https://github.com/daisukelab/dl-cliche.git@master --upgrade

2. Install [albumentations](https://github.com/albu/albumentations/).
3. Edit the `DATA_PATH` variable in `config.py` to the location where
you downloaded dataset copy from Kaggle.
4. Open and run `app/whale/Example_Humpback_Whale_Identification.ipynb`.

## Benefits and drawbacks

- Very simple design for both networks and training algorithm.
- All non-linearity can be learned by the model.
- Independent from model design, we can choose arbitrary networks best fit to the problem.
- Embeddings produced by the learnt model are simple data points in multi-dimensional Euclidean space where distances between data points are quite simply calculated.
- Training is easier comparing to Siamese networks for example.
- Less sensitive to class imbalance, training algorithm always picks equal number of samples from k-classes.
- Test time augmentation can be naturally applied for both getting prototypes and test samples' embeddings.

But

- Number of classes ProtoNets can train is mainly limited by memory size. Single GTX1080Ti can handle up to 20 classes for 1 shot for example.
- As far as I have tried, more k-way (k-classes) results in better performance, and it is limited by memory as written above.

## Towards better performance

- Augmentation matters.
- Image size also matters.
- TTA pushes score.
- and more...

## Resources

- Original paper: [Prototpyical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
(Snell et al).
- [Oscar Knagg](https://towardsdatascience.com/@oknagg)'s article: [Theory and concepts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
- [Oscar Knagg](https://towardsdatascience.com/@oknagg)'s article: [Discussion of implementation details](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d)
- [Radek Osmulski](https://www.kaggle.com/radek1)'s post on Kaggle discussion: [[LB 0.760] Fastai Starter Pack](https://www.kaggle.com/c/humpback-whale-identification/discussion/74647)
- [Radek Osmulski](https://medium.com/@radekosmulski)'s github repository: [Humpback Whale Identification Competition Starter Pack](https://github.com/radekosmulski/whale)