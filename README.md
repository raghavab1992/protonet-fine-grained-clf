# (Using) Prototypical Networks as a Fine Grained Classifier

This repository is heavily based on [Oscar Knagg](https://towardsdatascience.com/@oknagg)'s few-shot learning
implementation [github.com/oscarknagg/few-shot](https://github.com/oscarknagg/few-shot), focused on applying simple but strong [Prototpyical Networks](https://arxiv.org/pdf/1703.05175.pdf) to fine grained classification task.

Main contributions this repository provides:

- Practical application of few-shot machine learning system ready to real world fine-grained classification problems.
- Transfer learning ready to make quick training possible. Using ImageNet pre-trained models by default, or any networks even non-CNN are available.
- Proved in a fairly difficult Kaggle competition that ImageNet pretrained model works fine as core model of Prototypical Networks.

Unlike very clean original implementation, this repository contains
some dirty code to quickly present sample solution to a Kaggle competition
"[Humpback Whale Identification](https://www.kaggle.com/c/humpback-whale-identification/)".

Some of submission code borrows functions from [Radek Osmulski](https://medium.com/@radekosmulski)'s github repository.

I'd like to express sincere appreciation to both [Oscar Knagg](https://towardsdatascience.com/@oknagg) and [Radek Osmulski](https://medium.com/@radekosmulski). Thank you.

## Prototypical Networks

Prototypical Networks was proposed in the paper [Prototpyical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
(Snell et al), which calculates _prototype_ as a central point of class in Euclidean space, then test samples can be simply classified by measuring distances to the class prototypes.

In Prototypical Networks, model learns all the non-linearity. It encapsulates everything in between non-linear inputs and linear outputs, system design and training algorithm make it all possible.

![fig](assets/proto_nets_diagram.png)

Figure from original paper. Color circles: training samples, $c_i$: prototypes, $x$: test sample.

What Prototypical Networks scheme trains model is metrics in Euclidean space, this makes it quite handy tool for real world engineering.

Here's summary of nice traits for machine learning practitioners:

- Explainable: It discriminates classes in multi-dimensional Euclidean space, which many old fashioned engineers are familiar with. This is important so that we can explain to non-ML project stakeholders and finally bring the model to the real world projects. It’s not even cosine distance, just a conventional distance.
- Customizable: Any model can be used, so it is applicable to any problem; model is simply trained to map input data points to output data points in Euclidean space so that all classes can be distinguished by old fashioned distance.
- Few-shot ready: It works with long tail problems where very small number of samples are available with some classes, as well as imbalance of samples between classes. It is (almost as of now) proven in a Kaggle competition
"[Humpback Whale Identification](https://www.kaggle.com/c/humpback-whale-identification/)".
- Easy to train: (I think) this is almost free from difficult and computationally intensive hard mining that selects training samples to make it difficulter as training goes.

## Quick start

This project derives prerequisite below:

    This project is written in python 3.6 and Pytorch and assumes you have
    a GPU.

1. Install [dl-cliche from github](https://github.com/daisukelab/dl-cliche), excuse me this is my almost-private library to repeat cliche code.

    pip install git+https://github.com/daisukelab/dl-cliche.git@master --upgrade

2. Install [albumentations](https://github.com/albu/albumentations/).
3. Edit the `DATA_PATH` variable in `config.py` to the location where
you downloaded dataset copy from Kaggle.
4. Open and run `app/whale/Example_Humpback_Whale_Identification.ipynb` to reproduce whale identification solution.

## Benefits and drawbacks summary

- Very simple design for both networks and training algorithm.
- All non-linearity can be learned by the model.
- Independent from model design, we can choose arbitrary networks best fit to our problem.
- Embeddings produced by the learnt model are simple data points in multi-dimensional Euclidean space where distances between data points are quite simply calculated.
- Training is easier comparing to Siamese networks for example.
- Less sensitive to class imbalance, training algorithm always picks equal number of samples from k-classes.
- Test time augmentation can be naturally applied for both getting prototypes and test samples' embeddings.

But

- Number of classes ProtoNets can train is mainly limited by memory size. Single GTX1080Ti can handle up to 20 classes for 1 shot with 384x384 images for example.
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
