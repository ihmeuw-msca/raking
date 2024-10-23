What is raking?
===============

The objective of raking is to adjust the values in a contingency table so that they add up to known margins.

Raking with one categorical variable
------------------------------------

.. figure:: figures/raking_1D.png

    Raking with one categorical variable

In this case, there is only one margin and the observations must be adjusted to add up to the known margin. We want the raked values to respect the condition:

.. math::
    \sum_{i = 1}^I \beta_i = s

Raking with two categorical variables
-------------------------------------

.. figure:: figures/raking_2D.png

    Raking with two categorical variables

In this case, the margins are the sums over the rows and the sums over the columns and we want to adjust the observations in the table so they add up correctly to the known row sums and column sums. We want the raked values to respect the conditions:

.. math::
    \sum_{i = 1}^I \beta_{ij} = s^1_j \quad \forall j = 1 , \cdots , J

.. math::
    \sum_{j = 1}^J \beta_{ij} = s^2_i \quad \forall i = 1 , \cdots , I

Note that in this case, we need the margins to respect the condition:

.. math::
    \sum_{j = 1}^J s^1_j = \sum_{i = 1}^I s^2_i

otherwise the problem does not have a solution.

Raking with three categorical variables
---------------------------------------

.. figure:: figures/raking_3D.png

    Raking with three categorical variables

In this case, the margins are the sums over the three directions and we want to adjust the observations in the cubee so they add up correctly to the known sums. We want the raked values to respect the conditions:

.. math::
    \sum_{i = 1}^I \beta{ijk} = s^1_{jk} \quad \forall j = 1 , \cdots , J \text{ and } \forall k = 1 , \cdots , K

.. math::
    \sum_{j = 1}^J \beta{ijk} = s^2_{ik} \quad \forall i = 1 , \cdots , I \text{ and } \forall k = 1 , \cdots , K

.. math::
    \sum_{k = 1}^K \beta{ijk} = s^3_{ij} \quad \forall i = 1 , \cdots , I \text{ and } \forall j = 1 , \cdots , J

Note that in this case, we need the margins to respect the conditions:

.. math::
    \sum_{j = 1}^J s^1_{jk} = \sum_{i = 1}^I s^2_{ik} \quad \forall k = 1 , \cdots , K

.. math::
    \sum_{k = 1}^K s^2_{ik} = \sum_{j = 1}^J s^3_{ij} \quad \forall i = 1 , \cdots , I

.. math::
    \sum_{i = 1}^I s^3_{ij} = \sum_{k = 1}^K s^1_{jk} \quad \forall j = 1 , \cdots , J

otherwise the problem does not have a solution.
