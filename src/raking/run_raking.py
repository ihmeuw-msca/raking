"""Module to run the raking problems"""

import numpy as np
import pandas as pd

from raking.compute_constraints import (
    constraints_1D,
    constraints_2D,
    constraints_3D,
    constraints_USHD,
)
from raking.compute_covariance import check_covariance
from raking.formatting_methods import (
    format_data_1D,
    format_data_2D,
    format_data_3D,
    format_data_USHD,
)
from raking.raking_methods import (
    raking_chi2,
    raking_entropic,
    raking_general,
    raking_logit,
)
from raking.uncertainty_methods import compute_covariance, compute_gradient

pd.options.mode.chained_assignment = None


class RakingData:
    """Raking input data structure"""

    def __init__(
        self,
        df_obs: pd.DataFrame,
        df_margins: list,
        var_names: list = None,
        weights: str = None,
        lower: str = None,
        upper: str = None,
        use_case: str = None,
    ):
        """
        Constructor of RakingData.

        Parameters
        -----------
        df_obs : pd.DataFrame
            Observations data
        df_margins : list
            list of pd.DataFrame
        var_names : list of strings
            Names of the variables over which we rake (e.g. cause, race, county). None if using special case.
        weights : string
            Name of the column containing the raking weights
        lower : string
            Name of the column containing the lower boundaries (for logit raking)
        upper : string
            Name of the column containing the upper boundaries (for logit raking)
        use_case : string
            Name of the use case if using special case.
        """
        self.df_obs = df_obs
        self.df_margins = df_margins
        self.var_names = var_names
        self.weights = weights
        self.lower = lower
        self.upper = upper
        self.use_case = use_case
        self.check_input_init()

    def check_input_init(self):
        """Check inputs type and compatibility."""
        if self.use_case == None:
            assert isinstance(self.var_names, list), (
                "The variables over which we rake must be entered as a list."
            )
            assert len(self.var_names) in [1, 2, 3], (
                "The dimension of the raking problem must be 1, 2 or 3"
            )
            self.dim = len(self.var_names)
        else:
            assert self.use_case in ["USHD"], (
                "Only the special use case 'USHD' is currently implemented."
            )
            self.dim = "USHD"
            self.var_names = ["cause", "race", "county"]

        # Format the data for the raking
        if self.dim == 1:
            df_margins = self.df_margins[0]
            var_name = self.var_names[0]
            (y, s, I, q, l, h) = format_data_1D(
                self.df_obs,
                df_margins,
                var_name,
                self.weights,
                self.lower,
                self.upper,
            )
            self.y = y
            self.s = s
            self.I = I
            self.q = q
            self.l = l
            self.h = h
        elif self.dim == 2:
            df_margins_1 = self.df_margins[0]
            df_margins_2 = self.df_margins[1]
            (y, s1, s2, I, J, q, l, h) = format_data_2D(
                self.df_obs,
                df_margins_1,
                df_margins_2,
                self.var_names,
                self.weights,
                self.lower,
                self.upper,
            )
            self.y = y
            self.s1 = s1
            self.s2 = s2
            self.I = I
            self.J = J
            self.q = q
            self.l = l
            self.h = h
        elif self.dim == 3:
            df_margins_1 = self.df_margins[0]
            df_margins_2 = self.df_margins[1]
            df_margins_3 = self.df_margins[2]
            (y, s1, s2, s3, I, J, K, q, l, h) = format_data_3D(
                self.df_obs,
                df_margins_1,
                df_margins_2,
                df_margins_3,
                self.var_names,
                self.weights,
                self.lower,
                self.upper,
            )
            self.y = y
            self.s1 = s1
            self.s2 = s2
            self.s3 = s3
            self.I = I
            self.J = J
            self.K = K
            self.q = q
            self.l = l
            self.h = h
        elif self.dim == "USHD":
            df_margins = self.df_margins[0]
            (y, s, I, J, K, q, l, h) = format_data_USHD(
                self.df_obs, df_margins, self.weights, self.lower, self.upper
            )
            self.y = y
            self.s = s
            self.I = I
            self.J = J
            self.K = K
            self.q = q
            self.l = l
            self.h = h
        else:
            pass

    def rake(
        self,
        method: str = "chi2",
        alpha: float = 1,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        gamma0: float = 1.0,
        max_iter: int = 500,
    ):
        """
        Runs the raking.

        Parameters
        ----------
        method : string
            Name of the distance function used for the raking.
            Possible values are chi2, entropic, general, logit
        alpha : float
            Parameter of the distance function, alpha=1 is the chi2 distance, alpha=0 is the entropic distance
        rtol : float
            Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
        atol : float
            Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
        gamma0 : float
            Initial value for line search
        max_iter : int
            Number of iterations for Newton's root finding method
        """
        self.method = method
        assert isinstance(self.method, str), (
            "The name of the distance function used for the raking must be a string."
        )
        assert self.method in [
            "chi2",
            "entropic",
            "general",
            "logit",
        ], "The distance function must be chi2, entropic, general or logit."
        self.alpha = alpha
        self.rtol = rtol
        self.atol = atol
        self.gamma0 = gamma0
        self.max_iter = max_iter

        # Get the constraints for the raking
        if self.dim == 1:
            (A, s) = constraints_1D(self.s, self.I)
            self.A = A
            self.s = s
        elif self.dim == 2:
            (A, s) = constraints_2D(
                self.s1, self.s2, self.I, self.J, self.rtol, self.atol
            )
            self.A = A
            self.s = s
        elif self.dim == 3:
            (A, s) = constraints_3D(
                self.s1,
                self.s2,
                self.s3,
                self.I,
                self.J,
                self.K,
                self.rtol,
                self.atol,
            )
            self.A = A
            self.s = s
        elif self.dim == "USHD":
            (A, s) = constraints_USHD(
                self.s, self.I, self.J, self.K, self.rtol, self.atol
            )
            self.A = A
            self.s = s
        else:
            pass

        # Rake
        if self.method == "chi2":
            (beta, lambda_k) = raking_chi2(self.y, self.A, self.s, self.q)
            iter_eps = 1
        elif self.method == "entropic":
            (beta, lambda_k, iter_eps) = raking_entropic(
                self.y, self.A, self.s, self.q, self.gamma0, self.max_iter
            )
        elif self.method == "general":
            (beta, lambda_k, iter_eps) = raking_general(
                self.y,
                self.A,
                self.s,
                self.alpha,
                self.q,
                self.gamma0,
                self.max_iter,
            )
        elif self.method == "logit":
            (beta, lambda_k, iter_eps) = raking_logit(
                self.y,
                self.A,
                self.s,
                self.l,
                self.h,
                self.q,
                self.gamma0,
                self.max_iter,
            )
        else:
            pass

        # Create data frame for the raked values
        reverse_var = self.var_names.copy()
        reverse_var.reverse()
        self.df_obs.sort_values(by=reverse_var, inplace=True)
        self.df_obs["raked_value"] = beta

        # Keep track of the dual and the number of iterations
        self.beta = beta
        self.lambda_k = lambda_k
        self.num_iters = iter_eps

    def compute_variance(
        self,
        variance: str = None,
        sigma_yy: np.ndarray = None,
        sigma_ss: np.ndarray = None,
        sigma_ys: np.ndarray = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ):
        """
        Compute the covariance matrix.

        Parameters
        ----------
        variance: string
            Name of the column that contains the variance (if independence is assumed).
        sigma_yy: np.ndarray
            Covariance matrix of the observations. We assume that there are sorted by var3, var2, var1.
        sigma_ss: np.ndarray
            Covariance matrix of the margins.
        sigma_ys: np.ndarray
            Covariance matrix of the observations and the margins.
        rtol : float
            Relative tolerance to check whether the caovariance matrix is symmetric. See numpy.allclose documentation for details.
        atol : float
            Absolute tolerance to check whether the covariance matrix is symmetric. See numpy.allclose documentation for details.
        """
        if isinstance(variance, str):
            assert variance in self.df_obs.columns.tolist(), (
                "The column for the variance "
                + variance
                + " is missing from the observations data frame."
            )
            self.get_variance(variance)
        else:
            (sigma_yy, sigma_ss, sigma_ys) = check_covariance(
                sigma_yy, sigma_ss, sigma_ys, rtol, atol
            )
            self.sigma_yy = sigma_yy
            self.sigma_yy = sigma_yy
            self.sigma_yy = sigma_yy

        # Compute the covariance matrix of the raked values
        (Dphi_y, Dphi_s) = compute_gradient(
            self.beta,
            self.lambda_k,
            self.y,
            self.A,
            self.method,
            self.alpha,
            self.l,
            self.h,
            self.q,
        )
        sigma = compute_covariance(
            Dphi_y, Dphi_s, self.sigma_yy, self.sigma_ss, self.sigma_ys
        )
        self.df_obs["raked_variance"] = np.diag(sigma)
        self.Dphi_y = Dphi_y
        self.Dphi_s = Dphi_s
        self.sigma = sigma

    def get_variance(self, variance: str):
        """
        Get the variance columns of the observations and margins.

        Parameters
        ----------
        variance: string
            Name of the column that contains the variance (if independence is assumed).
        """
        sigma_yy = np.diag(self.df_obs[variance])
        for df_margin in self.df_margins:
            assert variance in df_margin.columns.tolist(), (
                "The column for the variance "
                + variance
                + " is missing from the margins data frame."
            )
        if self.dim == 1:
            sigma_ss = np.array([[self.df_margins[0][variance].iloc[0]]])
        elif self.dim == 2:
            variance1 = (
                self.df_margins[0]
                .sort_values(by=[self.var_names[1]])[variance]
                .to_numpy()
            )
            variance2 = (
                self.df_margins[1]
                .sort_values(by=[self.var_names[0]])[variance]
                .to_numpy()
            )
            sigma_ss = np.diag(
                np.concatenate([variance1, variance2[0 : (self.I - 1)]])
            )
        elif self.dim == 3:
            variance1 = (
                self.df_margins[0]
                .sort_values(by=[self.var_names[2], self.var_names[1]])[
                    variance
                ]
                .to_numpy()
                .reshape([self.J, self.K], order="F")
            )
            variance2 = (
                self.df_margins[1]
                .sort_values(by=[self.var_names[2], self.var_names[0]])[
                    variance
                ]
                .to_numpy()
                .reshape([self.I, self.K], order="F")
            )
            variance3 = (
                self.df_margins[2]
                .sort_values(by=[self.var_names[1], self.var_names[0]])[
                    variance
                ]
                .to_numpy()
                .reshape([self.I, self.J], order="F")
            )
            sigma_ss = np.diag(
                np.concatenate(
                    [
                        variance1[0 : (self.J - 1), 0 : self.K].flatten(
                            order="F"
                        ),
                        np.array([variance1[self.J - 1, self.K - 1]]),
                        variance2[0 : self.I, 0 : (self.K - 1)].flatten(
                            order="C"
                        ),
                        variance3[0 : (self.I - 1), 0 : self.J].flatten(
                            order="F"
                        ),
                    ]
                )
            )
        elif self.dim == "USHD":
            variance = (
                self.df_margins[0]
                .loc[self.df_margins[0].cause != "_all"]
                .sort_values(by=["cause"])[variance]
                .to_numpy()
            )
            sigma_ss = np.diag(
                np.concatenate(
                    [variance, np.zeros((self.I + self.J + 1) * self.K)]
                )
            )
        sigma_ys = np.zeros((np.shape(sigma_yy)[0], np.shape(sigma_ss)[0]))
        self.sigma_yy = sigma_yy
        self.sigma_ss = sigma_ss
        self.sigma_ys = sigma_ys
