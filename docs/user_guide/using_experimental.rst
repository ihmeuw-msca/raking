Using the experimental version of raking
========================================

This version allow more flexibility in the definition of the raking problem, including the inclusion of aggregates that must be raked together with the observations, and the possibility of having missing data. It does not include yet uncertainty propagation.

Input files format
------------------

You only need a single pandas data frame containing the information on both the observations and margins. This data frame must contain one column for each of the categorical variables (e.g. cause, location etc.) corresponding to the dimension of the problem, one column for the values of the observations being raked (e.g. number of death per cause and location), one column for the weights, and optionally one column for the lower bounds and one column for the upper bounds (if the raked values must stay bounded).

If the row corresponds to an observation present in the data frame, the weight must be strictly positive. If the row corresponds to a missing observation for which a raked value must be computed, the value must be set to np.nan and the weight must be set to 0. If the row corresponds to a constraint (either an observation with no uncertainty that must not move though the raking, or a known margin that must also stay constant), the weight must be set to np.inf. The most certain observations must be assigned bigger weights than the most uncertain observations.

Using the data builder
----------------------

You must first specify how the data are encoded in the pandas data frame. In dim_specs, you need to specify which value corresponds to the aggregate (e.g. if raking over cause, which value in the ``cause`` column corresponds to all cause). You also need to specify the name of the column that contain the values and the name of the column that contains the weights. For example, if you rake by cause and location, and all cause is encoded by math:`-1` and all location is encoded by math:`-1`:

.. code-block:: python
    data_builder = DataBuilder(
        dim_specs={'cause': -1, 'location': -1},
        value='value',
        weights='weights'
    )

You then need to build the data using your input pandas data frame:

.. code-block:: python
    data = data_builder.build(df)

Using the solver
----------------

You will need to specify which distance you want to use (see below) and enter the data you just build with the data builder:

.. code-block:: python
    solver = DualSolver(distance='entropic', data=data)

You then need to actually run the solver. It will return a pandas data frame that contains the column for the categorical variable and a column ``soln`` that contains the raked values. Note that only the observations are in the output, the aggregates are not included.

.. code-block:: python
    df_raked = solver.solve()

If you want more information on the convergence of the solver, you can type:

.. code-block:: python
    solver.result

Distance functions
------------------

The raking algorithm find the raked values that minimize the distance between the initial observations and the unknown raked values. In the package, you can choose between three distances. We denote math:`y_i` the initial observations and math:`\beta_i` the corresponding raked values. We assume that there are math:`N` observations.


Chi square distance
^^^^^^^^^^^^^^^^^^^

This is the fatest way the solve the raking problem, but we can obtained negative raked values.

.. math::
    f^{\chi^2} \left( \beta , y \right) = \sum_{i = 1}^N \frac{1}{2 y_i} \left( \beta_i - y_i \right)^2

Entropic distance
^^^^^^^^^^^^^^^^^

With this distance, the raked values have the same sign as the initial observations. This is useful when raking number of deaths for instance.

.. math::
    f^{\text{entropic}} \left( \beta , y \right) = \sum_{i = 1}^N \beta_i \log \left( \frac{\beta_i}{y_i} \right) - \beta_i + y_i

Logit distance
^^^^^^^^^^^^^^

With this distance, the raked values stay bounded between math:`l_i` and math:`h_i`. This is useful when raking prevalences for instance.

.. math::
    f^{\text{logit}} \left( \beta , y \right) = \sum_{i = 1}^N \left( \beta_i - l_i  \right) \log \frac{\beta_i - l_i}{y_i - l_i} + \left( h_i - \beta_i \right) \log \frac{h_i - \beta_i}{h_i - y_i}.

