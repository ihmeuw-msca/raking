Raking with uncertainty
=======================

In this case, we supposed that we have draws for the observations and the margins. We compute the raked values and the covariance matrix of the raked values

Raking function input
---------------------

These are the additional arguments needed by the raking function.

* cov_mat: Enter True otherwise to rake the mean of the draws and compute the covariance matrix.
* draws: Give the name of the column contatining the draws.
* sigma_yy: You can provide the covariance matrix of the observations. Otherwise, enter None and it will be computed using the draws.
* sigma_ss: You can provide the covariance matrix of the margins. Otherwise, enter None and it will be computed using the draws.
* sigma_ys: You can provide the covariance matrix of the observations and margins. Otherwise, enter None and it will be computed using the draws.

Raking function output
----------------------

The raking function will return a list of 4 variables:

* The first variable contains the observations data frame with an additional column "raked_value" that contains the raked values and another column called "variance" that contains the variances of the raked values.
* The second variable contains the gradient of the raked values with respect to the observations. This is mostly for development purposes.
* The third variable contains the gradient of the raked values with respect to the margins. This is mostly for development purposes.
* The fourth variable contains the covariance matrix of the raked values. The observations are ordered in the following order: If you have entered ["cause", "race", "county"] as your categorical variables and you have 2 causes, 2 races and 2 counties, the observations are ranked in the following order:

.. _tbl-grid:

+----------+--------+---------+------------+
|county    | race   | cause   | covariance |
|          |        |         |            |
+==========+========+=========+============+
| county 1 | race 1 | cause 1 | float      |
+----------+--------+---------+------------+
| county 1 | race 1 | cause 2 | float      |
+----------+--------+---------+------------+
| county 1 | race 2 | cause 1 | float      |
+----------+--------+---------+------------+
| county 1 | race 2 | cause 2 | float      |
+----------+--------+---------+------------+
| county 2 | race 1 | cause 1 | float      |
+----------+--------+---------+------------+
| county 2 | race 1 | cause 2 | float      |
+----------+--------+---------+------------+
| county 2 | race 2 | cause 1 | float      |
+----------+--------+---------+------------+
| county 2 | race 2 | cause 2 | float      |
+----------+--------+---------+------------+

