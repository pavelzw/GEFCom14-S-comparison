import numpy as np
from sklearn.neighbors import NearestNeighbors as nn_fun


def nnqf_filter(x_input,
                y_output,
                num_neighbors=10,
                q_quantile=0.5,
                var_weighting=True,
                minkowski_dist=2):
    """
    Parameters
    ----------
    x_input : numpy array ;
    Input matrix of dimension (N,S), with N representing the number of
    samples and S the number of features

    y_output : numpy array ;
    Output vector of dimension (N,)

    num_neighbors : int, default = 10 ;
    Number of nearest neighbors that the filter is going to search for

    q_quantile : float, default = 0.5 ;
    Must be a value between 0 and 1.
    Probability of the quantile that is going to be calculated from the
    nearest neighbors output values

    var_weighting : bool, default = True ;
    Value defining if the columns of the input matrix are going to be multiplied
    by the inverse of their variance

    minkowski_dist : int, default = 2 ;
    Parameter used to define the type of minkoswki distance used to calculate
    the nearest neighbors

    Returns
    -------
    yq_output : numpy array ;
    New output vector containing the quantiles of the output values of the
    input's nearest neighbors

    """
    # --
    # Each column of the input matrix is multiplied by the inverse of its variance,
    # in order to avoid a feature with a huge scale to overpower the others at the
    # moment of calculating the distances

    if var_weighting:
        var_weights = np.var(x_input, axis=0)
        x_input = var_weights ** (-1) * x_input

        # --
    # We calculate the nearest neighbor of each feature vector within the input matrix
    # and obtain their corresponding indices
    # The distance used is the minkowski distance with p = minkowski_dist

    x_neighbors = nn_fun(n_neighbors=num_neighbors, algorithm='auto', p=minkowski_dist).fit(x_input)
    dist, indx = x_neighbors.kneighbors(x_input)

    # --
    # We create a matrix containing the output values of nearest neighbors of
    # each input vector

    y_neighbors = y_output[indx[0, :]].T
    for i in range(1, np.size(x_input, 0)):
        values_to_add = y_output[indx[i, :]].T
        y_neighbors = np.vstack([y_neighbors, values_to_add])

    # --
    # We calculate the q_quantile of the nearest neighbors output values
    # and create with them a new output vector yq_output

    yq_output = np.quantile(y_neighbors, q=q_quantile, axis=1)

    return yq_output
