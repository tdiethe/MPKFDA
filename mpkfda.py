from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import check_array, column_or_1d, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel
import numpy as np
import scipy.sparse as sp
from scipy.signal import unit_impulse
import warnings


KERNELS = {
    'linear': linear_kernel,
    'poly': polynomial_kernel,
    'sigmoid': sigmoid_kernel,
    'rbf': rbf_kernel
}


# noinspection PyPep8Naming
class MPKFDA(BaseEstimator, ClassifierMixin):
    """
    Matching Pursuit Kernel Fisher Discriminant Analysis
    The multiclass support is handled according to a one-vs-one scheme.
    Parameters
    ----------
    k : int, optional(default=20)
        Number of basis functions to use
    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
        a callable.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        Current default is 'auto' which uses 1 / n_features,
        if ``gamma='scale'`` is passed then it uses 1 / (n_features * X.std())
        as value of gamma. The current default of gamma, 'auto', will change
        to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of
        'auto' is used as a default indicating that no explicit value of gamma
        was passed.
    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    tol: float, optional (default=0.01)
        Tolerance for monitoring convergence
    verbose : bool, default: False
        Enable verbose output.
    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> clf = MPKFDA(k=10)
    >>> clf.fit(X, y)
    MPKFDA(k=10, degree=3, gamma='auto', kernel='rbf', max_iter=-1, tol=0.001, verbose=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    def __init__(self, k, kernel='rbf', degree=3, gamma='auto', coef0=0.0, verbose=False, tol=0.01, set_to_zero=True):
        """
        Called when initializing the classifier
        """
        self.classes_ = None
        self.gamma = gamma
        self._gamma = None
        self.k = k
        self.kernel = kernel
        self.coef0 = coef0
        self._coef0 = None
        self._degree = degree
        self._verbose = verbose
        self._sparse = None
        self.shape_fit_ = None
        if tol is None:
            raise ValueError
        self._tol = tol
        self.set_to_zero = set_to_zero

        self.intercept_ = None
        self._intercept_ = None
        self.coef_ = None
        self._coef_ = None
        self.__Xfit = None
        self.R_ = None
        self.support_ = None

    def _validate_targets(self, y):
        """Validation of y and class_weight.
        Default implementation for SVR and one-class; overridden in BaseSVC.
        """
        # XXX this is ugly.
        # Regression models should not have a class_weight_ attribute.
        self.class_weight_ = np.empty(0)
        return column_or_1d(y, warn=True).astype(np.float64)

    def _validate_for_predict(self, X):
        check_is_fitted(self, 'support_')

        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)
        m, n = X.shape

        if self.kernel == "precomputed":
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))
        elif n != self.shape_fit_[1]:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time" %
                             (n, self.shape_fit_[1]))
        return X

    @property
    def n_support_(self):
        if self.support_ is None:
            return None
        return len(self.support_)

    def fit(self, X, y):
        """Fit the SVM model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).
        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)
        Returns
        -------
        self : object
        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.
        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        sparse = sp.isspmatrix(X)
        if sparse and self.kernel == "precomputed":
            raise TypeError("Sparse precomputed kernels are not supported.")
        self._sparse = sparse and not callable(self.kernel)

        # if self.k > X.shape[1]:
        #     raise ValueError("Selected k is greater than the input dimension - this will cause errors")

        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        y = self._validate_targets(y)

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        kernel_uses_gamma = (not callable(self.kernel) and self.kernel
                             not in ('linear', 'precomputed'))
        kernel_uses_coef0 = (not callable(self.kernel) and self.kernel == 'polynomial')

        if kernel_uses_gamma:
            if self.gamma in ('scale', 'auto_deprecated'):
                if sparse:
                    # std = sqrt(E[X^2] - E[X]^2)
                    X_std = np.sqrt((X.multiply(X)).mean() - (X.mean()) ** 2)
                else:
                    X_std = X.std()
                if self.gamma == 'scale':
                    if X_std != 0:
                        self._gamma = 1.0 / (X.shape[1] * X_std)
                    else:
                        self._gamma = 1.0
                else:
                    if kernel_uses_gamma and not np.isclose(X_std, 1.0):
                        # NOTE: when deprecation ends we need to remove explicitly
                        # setting `gamma` in examples (also in tests). See
                        # https://github.com/scikit-learn/scikit-learn/pull/10331
                        # for the examples/tests that need to be reverted.
                        warnings.warn("The default value of gamma will change "
                                      "from 'auto' to 'scale' in version 0.22 to "
                                      "account better for unscaled features. Set "
                                      "gamma explicitly to 'auto' or 'scale' to "
                                      "avoid this warning.", FutureWarning)
                    self._gamma = 1.0 / X.shape[1]
            elif self.gamma == 'auto':
                self._gamma = 1.0 / X.shape[1]
            else:
                self._gamma = self.gamma
        else:
            self._gamma = None

        if kernel_uses_coef0:
            self._coef0 = self.coef0
        else:
            self._coef0 = None

        self.shape_fit_ = X.shape

        kernel = self.kernel if callable(self.kernel) else KERNELS[self.kernel]

        if callable(kernel):
            # kernel = 'precomputed'
            self.__Xfit = X
            K = self._compute_kernel(X, kernel)

            if K.shape[0] != K.shape[1]:
                raise ValueError("K.shape[0] should be equal to K.shape[1]")
        else:
            raise ValueError("Unknown kernel type")

        B = get_B(y)
        e = np.zeros(self.k, dtype=int)
        selected = set()

        # plot_matrix(K, "K")

        K_hat = K
        maxJ = np.zeros(self.k)

        for k in range(self.k):
            J = np.zeros(len(y))
            K_hat_y = np.dot(K_hat.T, 2 * (y - 0.5))
            kBk = np.dot(np.dot(K_hat.T, B), K_hat)
            for i in range(len(y)):
                if i in selected:
                    continue
                kiy = K_hat_y[i]
                ei = unit_impulse(len(y), i)
                kiBki = np.dot(np.dot(ei.T, kBk), ei)
                J[i] = (kiy ** 2) / kiBki
            ek = np.argmax(J)
            if ek in selected:
                break
                # continue
            e[k] = ek
            if self._verbose:
                print("Iteration {:2d}, selected {:2d},  maximum of objective {:.4f}".format(k, e[k], J[e[k]]))
            maxJ[k] = J[e[k]]
            diff = maxJ[k - 1] - maxJ[k]
            if k > 0 > diff:
                if self._verbose:
                    print("max(J) increased!")
                # raise ValueError("Objective increased")
            if k > 0 and 0 < diff < self._tol:
                break
                # pass

            selected.add(e[k])
            K_hat = deflate(K_hat, e[k], set_to_zero=self.set_to_zero)

            # plot_matrix(K_hat, "K_hat {}".format(k))

        # noinspection PyUnboundLocalVariable
        self.support_ = e[:k+1]

        self.R_ = compute_projection_matrix(K, self.support_, verbose=self._verbose)

        # self.R_ = np.linalg.inv(np.linalg.cholesky(Kii + np.eye(self.n_support_) * 1e-100))

        P = np.dot(K[:, self.support_], self.R_)

        # Define the new data matrix using the support (for prediction time)
        # self.__Xfit = K[self.support_, :]

        self.coef_, self.intercept_ = lda(P, y)
        if self._verbose:
            print()

        # noinspection PyAttributeOutsideInit
        self.fit_status_ = 1
        return self

    def predict(self, X):
        """Perform regression on samples in X.
        For an one-class model, +1 (inlier) or -1 (outlier) is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        return (self.decision_function(X) > 0).astype(float)

    def _compute_kernel(self, X, kernel):
        """Return the data transformed by a callable kernel"""
        if callable(kernel):
            # in the case of precomputed kernel given as a function, we
            # have to compute explicitly the kernel matrix
            kwargs = {}
            if self._coef0 is not None:
                kwargs['coef0'] = self._coef0
            if self._gamma is not None:
                kwargs['gamma'] = self._gamma
            K = kernel(X, self.__Xfit, **kwargs)
            if sp.issparse(K):
                K = K.toarray()
            X = np.asarray(K, dtype=np.float64, order='C')
        return X

    def decision_function(self, X):
        X = self._validate_for_predict(X)
        if X.ndim == 1:
            X = check_array(X, order='C')

        kernel = self.kernel if callable(self.kernel) else KERNELS[self.kernel]
        if callable(self.kernel):
            kernel = 'precomputed'
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))

        K = self._compute_kernel(X, kernel)
        P = np.dot(K[:, self.support_], self.R_)
        return np.dot(P, self.coef_) + self.intercept_

    # def predict_proba(self, X):
    #     """Compute probabilities of possible outcomes for samples in X.
    #     The model need to have probability information computed at training
    #     time: fit with attribute `probability` set to True.
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_features)
    #         For kernel="precomputed", the expected shape of X is
    #         [n_samples_test, n_samples_train]
    #     Returns
    #     -------
    #     T : array-like, shape (n_samples, n_classes)
    #         Returns the probability of the sample for each class in
    #         the model. The columns correspond to the classes in sorted
    #         order, as they appear in the attribute `classes_`.
    #     Notes
    #     -----
    #     The probability model is created using cross validation, so
    #     the results can be slightly different than those obtained by
    #     predict. Also, it will produce meaningless results on very small
    #     datasets.
    #     """
    #     probs = np.clip(self.decision_function(X), 0, 1)
    #     return np.stack([1 - probs, probs], axis=1)


# noinspection PyPep8Naming
def deflate(K, i, set_to_zero=False):
    # Deflate using orthogonalized Hotelling's method
    tau = K[i, :]
    # ensure tau is norm 1
    tau = tau / np.linalg.norm(tau)
    ttp = np.outer(tau, tau)
    tpt = np.inner(tau, tau)
    # K = np.dot(np.eye(m) - (ttp / tpt), K)
    K = K - np.dot(np.dot(ttp, K), tpt)
    if set_to_zero:
        # Optionally set the vector to zero
        K[i, :] = 0
    return K


# noinspection PyPep8Naming
def compute_projection_matrix(K, alpha, verbose=False):
    Kii = K[np.ix_(alpha, alpha)]

    try:
        Kii_inv = np.linalg.inv(Kii)
    except np.linalg.LinAlgError:
        if verbose:
            print("Inverse K_ii failed, adding ridge")
        Kii_inv = np.linalg.inv(Kii + np.eye(len(alpha)) * 1e-10)

    try:
        R = np.linalg.cholesky(Kii_inv)
    except np.linalg.LinAlgError:
        if verbose:
            print("Cholesky K_ii_inv failed, falling back to QR")
        # self.R_ = np.linalg.cholesky(Kii_inv + np.eye(self.n_support_) * 1e-10)
        R, _ = np.linalg.qr(Kii_inv)
    return R


# noinspection PyPep8Naming
def get_B(y):
    yp = y == 1
    yn = y == 0

    mp = sum(yp)
    mn = sum(yn)
    m = mp + mn

    if mn == mp:
        D = np.eye(m)
    else:
        D = np.diag((2 * mn / m * yp) + (2 * mp / m * yn))

    yp_outer = np.outer(yp, yp)
    yn_outer = np.outer(yn, yn)

    Cp = yp_outer.astype(float)
    Cn = yn_outer.astype(float)

    if mn != mp:
        Cp *= 2 * mn / (m * mp)
        Cn *= 2 * mp / (m * mn)

    B = D - Cp - Cn
    return B


# noinspection PyPep8Naming
def lda(X, y):
    yp = y == 1
    yn = y == 0
    Xp = X[yp, :]
    Xn = X[yn, :]
    Cp = np.dot(Xp.T, Xp)
    Cn = np.dot(Xn.T, Xn)
    mp = np.mean(Xp, axis=0)
    mn = np.mean(Xn, axis=0)
    w = np.linalg.solve(Cp + Cn, mp - mn)
    b = -0.5 * (np.dot(w.T, mp) + np.dot(w.T, mn))
    return w, b


# noinspection PyPep8Naming,PyUnreachableCode
def lda2(X, y):
    yp = y == 1
    yn = y == 0
    Xm = X.mean(axis=0)
    Xp = X[yp, :]
    Xn = X[yn, :]
    mp = Xp.mean(axis=0)
    mn = Xn.mean(axis=0)

    # Scatter within
    Sw = np.cov(Xp.T) + np.cov(Xn.T)

    # Scatter between
    Xm_mp = Xm - mp
    Xm_mn = Xm - mn
    Sb = np.outer(Xm_mn, Xm_mn) + np.outer(Xm_mp, Xm_mp)

    # Method using the square root of scatter between - see Max Welling's notes:
    # https://www.ics.uci.edu/~welling/teaching/273ASpring09/Fisher-LDA.pdf
    if False:
        U, V, W = np.linalg.svd(Sb)
        Sb_sqr = np.dot(np.dot(U, np.diag(np.sqrt(V))), W)
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.solve(Sb_sqr, Sw), Sb_sqr))

    else:
        # Compute standard Fisher in reduced space
        try:
            eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(Sw, Sb))
        except np.linalg.LinAlgError:
            raise ValueError("Scatter matrix is singular, try increasing the tolerance")

    w = eigenvectors[np.argmax(eigenvalues), :]
    return w, 0


# noinspection PyPep8Naming
def plot_matrix(A, name=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    # print(fig.number)
    ax = fig.subplots(1, 1)
    ax.matshow(A)
    # plt.show()
    plt.draw()
    if name:
        plt.savefig(name)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Binary
    from sklearn.datasets import make_classification
    n_samples = 100
    data = make_classification(n_samples=n_samples, n_features=50, n_informative=20, n_redundant=0, random_state=42)
    clf = MPKFDA(k=5, gamma='scale', verbose=True)
    X_train, X_test, y_train, y_test = train_test_split(*data, test_size=.33, random_state=42)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    # prob_pos_clf = clf.predict_proba(X_test)

    # print('\n'.join("true=%d, pred=%d" % (y_i, y_i_hat) for (y_i, y_i_hat) in zip(y_test, y_hat)))
    # clf_score = brier_score_loss(y_test, prob_pos_clf)
    # print("No calibration: %1.3f" % clf_score)
    print('Error rate = %.4f' % np.mean([0.0 if int(y_i) == int(y_i_hat) else 1.0 for (y_i, y_i_hat)
                                        in zip(y_test.reshape(-1).tolist(), y_hat)]))

    # Multi-class
    from sklearn.datasets import make_blobs

    n_samples = 900
    centers = [(-5, -5), (0, 0), (5, 5)]
    data = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)

    clf = OneVsRestClassifier(MPKFDA(k=2, tol=0.01))
    # y[:n_samples / 3] = 0
    # y[n_samples / 3:n_samples * 2 / 3] = 1
    # y[n_samples * 2 / 3:] = 4
    X_train, X_test, y_train, y_test = train_test_split(*data, test_size=.33, random_state=42)

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    # prob_pos_clf = clf.predict_proba(X_test)

    # print('\n'.join("true=%d, pred=%d" % (y_i, y_i_hat) for (y_i, y_i_hat) in zip(y_test, y_hat)))
    # clf_score = brier_score_loss(y_test, prob_pos_clf)
    # print("No calibration: %1.3f" % clf_score)
    print('Error rate = %.4f' % np.mean([0.0 if int(y_i) == int(y_i_hat) else 1.0 for (y_i, y_i_hat)
                                        in zip(y_test.reshape(-1).tolist(), y_hat)]))
