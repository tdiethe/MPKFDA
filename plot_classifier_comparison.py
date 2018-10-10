#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Modified to include MPKFDA by Tom Diethe
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mpkfda import MPKFDA
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context('paper')

print(__doc__)

h = .02  # step size in the mesh

tol = 1e-5
set_to_zero = True
verbose = False

classifiers = (
    # ('KNN (3)',         KNeighborsClassifier(3)),
    ('Linear SVM',      SVC(kernel="linear", C=0.025)),
    ('RBF SVM',         SVC(gamma=2, C=1)),
    # ('Decision Tree',   DecisionTreeClassifier(max_depth=5)),
    # ('Random Forest',   RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    # ('AdaBoost',        AdaBoostClassifier()),
    # ('Naive Bayes',     GaussianNB()),
    # ('FDA',             LinearDiscriminantAnalysis()),
    # ('QDA',             QuadraticDiscriminantAnalysis()),
    ('Linear MPKFDA',  MPKFDA(kernel="linear", k=2, verbose=verbose, tol=tol, set_to_zero=set_to_zero)),
    ('Linear MPKFDA', MPKFDA(kernel="linear", k=10, verbose=verbose, tol=tol, set_to_zero=set_to_zero)),
    ('RBF MPKFDA',      MPKFDA(gamma=2, k=10, verbose=verbose, tol=tol, set_to_zero=set_to_zero)),
    ('RBF MPKFDA', MPKFDA(gamma=2, k=20, verbose=verbose, tol=tol, set_to_zero=set_to_zero)),
    ('RBF MPKFDA', MPKFDA(gamma=2, k=40, verbose=verbose, tol=tol, set_to_zero=set_to_zero)),
    # ('RBF MPKFDA 20', MPKFDA(gamma=2, k=20, verbose=True, tol=0, set_to_zero=True)),
    # ('RBF MPKFDA 20', MPKFDA(gamma=2, k=20, verbose=True, tol=0, set_to_zero=False)),
    # ('RBF MPKFDA 20', MPKFDA(gamma=2, k=20, verbose=True, tol=tol, set_to_zero=True)),
    # ('RBF MPKFDA 20', MPKFDA(gamma=2, k=20, verbose=True, tol=tol, set_to_zero=False)),
    # ('RBF MPKFDA 20', MPKFDA(gamma=2, k=20, verbose=True, tol=1e-3, set_to_zero=True)),
    # ('RBF MPKFDA 20', MPKFDA(gamma=2, k=20, verbose=True, tol=1e-3, set_to_zero=False)),
)

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)
rng = np.random.RandomState(42)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = {
    "moons": make_moons(noise=0.3, random_state=42),
    "circles": make_circles(noise=0.2, factor=0.5, random_state=42),
    "linsep": linearly_separable
}

plt.ion()
fig = plt.figure(1, figsize=(3 * (1 + len(classifiers)), 3 * len(datasets)))
axs = fig.subplots(nrows=len(datasets), ncols=len(classifiers) + 1)

# iterate over datasets
for i, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = datasets[ds]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    # cm = sns.cm.mpl_cm
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # ax = figure.add_subplot(len(datasets), len(classifiers) + 1, i)

    plt.figure(fig.number)
    # axs = fig.gca()
    if len(datasets) == 1:
        ax = axs[0]
    else:
        ax = axs[i, 0]
    plt.subplot(ax)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    # plt.show()
    # plt.draw()

    # iterate over classifiers
    for j, (name, clf) in enumerate(classifiers):
        # ax = figure.add_subplot(len(datasets), len(classifiers) + 1, i)

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # May have been plotting inside fit (!?) so switch back to current figure
        plt.figure(fig.number)
        if len(datasets) == 1:
            ax = axs[j + 1]
        else:
            ax = axs[i, j + 1]
        plt.subplot(ax)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, 51, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        title = "{} sparsity={}".format(name, clf.n_support_) if hasattr(clf, 'n_support_') else name
        ax.set_title(title)
        ax.text(xx.max() - .3, yy.min() + .3,
                ('{:.2f}'.format(score)).lstrip('0'), size=15, horizontalalignment='right')
        # plt.show()
        # plt.draw()

# figure.subplots_adjust(left=.02, right=.98)
fig.subplots_adjust(left=.02, right=.98)
plt.savefig('classifier_comparison.png')
plt.show()
