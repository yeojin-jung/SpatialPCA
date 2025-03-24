import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
import networkx as nx

import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

from utils import *
from graphSVD import *

class GpLSI_(object):
    def __init__(
        self,
        lambd=None,
        lamb_start=0.0001,
        step_size=1.2,
        grid_len=29,
        maxiter=50,
        eps=1e-05,
        method="two-step",
        use_mpi=False,
        verbose=0,
        precondition=False,
        initialize=True,
        fast_option=False,
        sparsity=False
    ):
        """
        Parameters
        -----------

        """
        self.lambd = lambd
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.maxiter = maxiter
        self.eps = eps
        self.method = method
        self.verbose = verbose
        self.use_mpi = use_mpi
        self.precondition = precondition
        self.initialize = initialize
        self.fast_option = fast_option
        self.sparsity = sparsity

    def fit(self, X, K, edge_df, weights):
        if self.method == "pLSI":
            print("Running pLSI...")
            self.U, self.L, self.V = svds(X, k=K)
            self.L = np.diag(self.L)
            self.V = self.V.T
            self.U_init = None
        else:
            print("Running graph aligned pLSI...")
            (
                self.U,
                self.V,
                self.L,
                self.U_init,
                self.V_init,
                self.L_init,
                self.lambd,
                self.lambd_errs,
                self.beta_cv,
                self.beta_errs,
                self.used_iters
            ) = graphSVD(
                X,
                K,
                edge_df,
                weights,
                self.lamb_start,
                self.step_size,
                self.grid_len,
                self.maxiter,
                self.eps,
                self.verbose,
                self.initialize,
                self.fast_option,
                self.sparsity
            )