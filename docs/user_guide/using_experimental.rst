Using the experimental version of raking
========================================

This version allow more flexibility in the definition of the raking problem, including the inclusion of aggregates that must be raked together with the observations, and the possibility of having missing data. It does not include yet uncertainty propagation.

Input files format
------------------

You only need a single pandas data frame containing the information on both the observations and margins. This data frame must contain one column for each of the categorical variables (e.g. cause, location etc.) corresponding to the dimension of the problem, one column for the values of the observations being raked (e.g. number of death per cause and location), one column for the weights, and optionally one column for the lower bounds and one column for the upper bounds (if the raked values must stay bounded).

If the row corresponds to an observation present in the data frame, the weight must be strictly positive. If the row corresponds to a missing observation for which a raked value must be computed, the value must be set to np.nan and the weight must be set to 0. If the row corresponds to a constraint (either an observation with no uncertainty that must not move though the raking, or a known margin that must also stay constant), the weight must be set to np.inf. The most certain observations must be assigned bigger weights than the most uncertain observations.

1D raking
^^^^^^^^^

The input data frame looks like this:

========  ======  =======
  value    var1   weights
========  ======  =======
2.647452     0      1.0
2.216220     1      1.0
2.077133     2      1.0
6.947722    -1      inf

2D raking
^^^^^^^^^

The input data frame looks like this:

=========  ======  ======  =======
  value     var1    var2   weights
=========  ======  ======  =======
 2.563735     0      0       1.0
 2.858330     1      0       1.0
 2.784224     2      0       1.0
 2.310950     0      1       1.0
 2.710887     1      1       1.0
 1.989885     2      1       1.0
 2.177620     0      2       1.0
 2.662977     1      2       1.0
 2.892555     2      2       1.0
 2.106875     0      3       1.0
 2.553026     1      3       1.0
 1.959236     2      3       1.0
 2.721098     0      4       1.0
 2.889300     1      4       1.0
 2.751675     2      4       1.0
 8.365571    -1      0       inf
 6.879161    -1      1       inf
 7.627874    -1      2       inf
 6.593738    -1      3       inf
 8.477998    -1      4       inf
11.777520     0     -1       inf
13.727585     1     -1       inf
12.439237     2     -1       inf

3D raking
^^^^^^^^^

The input data frame looks like this:

=========  ======  ======  ======  =======
  value     var1    var2    var3   weights
=========  ======  ======  ======  =======
 2.593318     0       0       0      1.0
 1.911339     1       0       0      1.0
 2.745467     2       0       0      1.0
 2.863165     0       1       0      1.0
 2.416575     1       1       0      1.0
 2.864417     2       1       0      1.0
 2.974201     0       2       0      1.0
 2.820483     1       2       0      1.0
 2.850434     2       2       0      1.0
 1.955305     0       3       0      1.0
 2.315445     1       3       0      1.0
 2.218648     2       3       0      1.0
 2.370183     0       0       1      1.0
 2.062493     1       0       1      1.0
 2.504071     2       0       1      1.0
 2.474593     0       1       1      1.0
 2.931057     1       1       1      1.0
 2.320663     2       1       1      1.0
 2.007644     0       2       1      1.0
 2.589161     1       2       1      1.0
 2.258221     2       2       1      1.0
 2.799860     0       3       1      1.0
 2.448079     1       3       1      1.0
 2.341225     2       3       1      1.0
 1.913206     0       0       2      1.0
 2.733666     1       0       2      1.0
 2.652417     2       0       2      1.0
 2.858952     0       1       2      1.0
 2.905375     1       1       2      1.0
 2.792098     2       1       2      1.0
 2.828666     0       2       2      1.0
 2.292537     1       2       2      1.0
 2.833206     2       2       2      1.0
 2.467913     0       3       2      1.0
 2.914466     1       3       2      1.0
 2.899470     2       3       2      1.0
 2.032629     0       0       3      1.0
 2.588637     1       0       3      1.0
 2.203789     2       0       3      1.0
 2.403473     0       1       3      1.0
 2.735810     1       1       3      1.0
 2.182987     2       1       3      1.0
 2.017156     0       2       3      1.0
 2.614052     1       2       3      1.0
 2.145872     2       2       3      1.0
 2.171672     0       3       3      1.0
 2.862737     1       3       3      1.0
 2.292744     2       3       3      1.0
 2.588256     0       0       4      1.0
 2.654022     1       0       4      1.0
 2.333455     2       0       4      1.0
 2.946000     0       1       4      1.0
 2.642889     1       1       4      1.0
 2.852856     2       1       4      1.0
 2.799073     0       2       4      1.0
 2.449517     1       2       4      1.0
 2.478215     2       2       4      1.0
 2.495297     0       3       4      1.0
 2.442168     1       3       4      1.0
 2.168508     2       3       4      1.0
 7.236811    -1       0       0      inf
 8.186707    -1       1       0      inf
 8.291399    -1       2       0      inf
 6.636177    -1       3       0      inf
 6.715939    -1       0       1      inf
 7.831003    -1       1       1      inf
 6.631029    -1       2       1      inf
 7.799354    -1       3       1      inf
 7.305898    -1       0       2      inf
 8.333519    -1       1       2      inf
 7.868985    -1       2       2      inf
 8.227273    -1       3       2      inf
 7.001628    -1       0       3      inf
 7.313182    -1       1       3      inf
 6.813642    -1       2       3      inf
 7.464398    -1       3       3      inf
 7.820274    -1       0       4      inf
 8.418176    -1       1       4      inf
 7.591127    -1       2       4      inf
 6.832504    -1       3       4      inf
10.541226     0      -1       0      inf
 9.410686     1      -1       0      inf
10.399182     2      -1       0      inf
 9.742340     0      -1       1      inf
 9.996250     1      -1       1      inf
 9.238736     2      -1       1      inf
10.169336     0      -1       2      inf
10.676044     1      -1       2      inf
10.890296     2      -1       2      inf
 8.893450     0      -1       3      inf
10.988263     1      -1       3      inf
 8.711137     2      -1       3      inf
10.900685     0      -1       4      inf
10.148994     1      -1       4      inf
 9.612402     2      -1       4      inf
11.777520     0       0      -1      inf
12.085802     1       0      -1      inf
12.217229     2       0      -1      inf
13.727585     0       1      -1      inf
13.697724     1       1      -1      inf
12.657279     2       1      -1      inf
12.439237     0       2      -1      inf
12.459307     1       2      -1      inf
12.297637     2       2      -1      inf
12.302695     0       3      -1      inf
12.977404     1       3      -1      inf
11.679607     2       3      -1      inf

USHD raking (level 1 to all-cause)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input data frame looks like this:

=========  =====  ====  ======  =======
  value    cause  race  county  weights
=========  =====  ====  ======  =======
 7.619521   -1     -1    301      1.0
 4.612771    1     -1    301      1.0
 0.527299    2     -1    301      1.0
 2.413023    3     -1    301      1.0
 0.758506   -1      2    301      1.0
 0.512337    1      2    301      1.0
 0.043705    2      2    301      1.0
 0.227653    3      2    301      1.0
 3.878259   -1      4    301      1.0
 2.580850    1      4    301      1.0
 0.251599    2      4    301      1.0
 1.182016    3      4    301      1.0
 3.196077   -1      5    301      1.0
 1.691201    1      5    301      1.0
 0.254102    2      5    301      1.0
 1.171041    3      5    301      1.0
 0.007639   -1      6    301      1.0
 0.003628    1      6    301      1.0
 0.000683    2      6    301      1.0
 0.003480    3      6    301      1.0
 0.135695   -1      7    301      1.0
 0.103740    1      7    301      1.0
 0.005382    2      7    301      1.0
 0.041140    3      7    301      1.0
24.251880   -1     -1    302      1.0
18.114488    1     -1    302      1.0
 1.458384    2     -1    302      1.0
 5.411759    3     -1    302      1.0
 3.217496   -1      2    302      1.0
 1.944748    1      2    302      1.0
 0.155655    2      2    302      1.0
 0.891519    3      2    302      1.0
13.427143   -1      4    302      1.0
 9.477908    1      4    302      1.0
 1.032675    2      4    302      1.0
 2.520661    3      4    302      1.0
 6.894521   -1      5    302      1.0
 5.039603    1      5    302      1.0
 0.369887    2      5    302      1.0
 2.076813    3      5    302      1.0
 0.025593   -1      6    302      1.0
 0.016065    1      6    302      1.0
 0.002706    2      6    302      1.0
 0.012746    3      6    302      1.0
 1.708524   -1      7    302      1.0
 1.207481    1      7    302      1.0
 0.073614    2      7    302      1.0
 0.440330    3      7    302      1.0
 7.240533   -1     -1    303      1.0
 4.236110    1     -1    303      1.0
 0.598651    2     -1    303      1.0
 2.328931    3     -1    303      1.0
 1.167699   -1      2    303      1.0
 0.749765    1      2    303      1.0
 0.085868    2      2    303      1.0
 0.394253    3      2    303      1.0
 3.432895   -1      4    303      1.0
 1.975344    1      4    303      1.0
 0.273134    2      4    303      1.0
 0.906347    3      4    303      1.0
 2.899930   -1      5    303      1.0
 1.766393    1      5    303      1.0
 0.247110    2      5    303      1.0
 1.106343    3      5    303      1.0
 0.005331   -1      6    303      1.0
 0.002848    1      6    303      1.0
 0.000520    2      6    303      1.0
 0.002428    3      6    303      1.0
 0.131920   -1      7    303      1.0
 0.076260    1      7    303      1.0
 0.004744    2      7    303      1.0
 0.037682    3      7    303      1.0
43.886284   -1     -1     -1      inf
27.823886    1     -1     -1      inf
 3.107935    2     -1     -1      inf
12.954463    3     -1     -1      inf

USHD raking (level 2 to level 1 parent cause)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input data frame looks like this:

=========  =====  ====  ======  =======
  value    cause  race  county  weights
=========  =====  ====  ======  =======
 0.118175    1     -1    301      1.0
 0.293406    2     -1    301      1.0
 0.033900    3     -1    301      1.0
 0.010369    1      2    301      1.0
 0.027488    2      2    301      1.0
 0.003967    3      2    301      1.0
 0.049080    1      4    301      1.0
 0.209313    2      4    301      1.0
 0.015435    3      4    301      1.0
 0.053521    1      5    301      1.0
 0.135591    2      5    301      1.0
 0.017261    3      5    301      1.0
 0.000157    1      6    301      1.0
 0.000570    2      6    301      1.0
 0.000085    3      6    301      1.0
 0.001131    1      7    301      1.0
 0.003324    2      7    301      1.0
 0.000431    3      7    301      1.0
 0.376631    1     -1    302      1.0
 0.820883    2     -1    302      1.0
 0.073213    3     -1    302      1.0
 0.037681    1      2    302      1.0
 0.097146    2      2    302      1.0
 0.010592    3      2    302      1.0
 0.238464    1      4    302      1.0
 0.634665    2      4    302      1.0
 0.043052    3      4    302      1.0
 0.087516    1      5    302      1.0
 0.263410    2      5    302      1.0
 0.021672    3      5    302      1.0
 0.000758    1      6    302      1.0
 0.002557    2      6    302      1.0
 0.000289    3      6    302      1.0
 0.015757    1      7    302      1.0
 0.045956    2      7    302      1.0
 0.006432    3      7    302      1.0
 0.103712    1     -1    303      1.0
 0.418953    2     -1    303      1.0
 0.041973    3     -1    303      1.0
 0.013333    1      2    303      1.0
 0.045058    2      2    303      1.0
 0.011201    3      2    303      1.0
 0.035000    1      4    303      1.0
 0.224885    2      4    303      1.0
 0.016696    3      4    303      1.0
 0.046943    1      5    303      1.0
 0.175575    2      5    303      1.0
 0.016886    3      5    303      1.0
 0.000111    1      6    303      1.0
 0.000377    2      6    303      1.0
 0.000065    3      6    303      1.0
 0.000934    1      7    303      1.0
 0.002890    2      7    303      1.0
 0.000409    3      7    303      1.0
 0.604437    1     -1     -1      inf
 2.273448    2     -1     -1      inf
 0.230049    3     -1     -1      inf
 0.624089   -1     -1    301      inf
 1.784524   -1     -1    302      inf
 0.699322   -1     -1    303      inf
 0.048717   -1      2    301      inf
 0.280005   -1      4    301      inf
 0.288797   -1      5    301      inf
 0.000756   -1      6    301      inf
 0.005816   -1      7    301      inf
 0.174820   -1      2    302      inf
 1.142346   -1      4    302      inf
 0.385038   -1      5    302      inf
 0.002607   -1      6    302      inf
 0.079712   -1      7    302      inf
 0.095836   -1      2    303      inf
 0.325318   -1      4    303      inf
 0.271897   -1      5    303      inf
 0.000565   -1      6    303      inf
 0.005705   -1      7    303      inf

Using the data builder
----------------------

You must first specify how the data are encoded in the pandas data frame. In dim_specs, you need to specify which value corresponds to the aggregate (e.g. if raking over cause, which value in the ``cause`` column corresponds to all cause). You also need to specify the name of the column that contain the values and the name of the column that contains the weights. 

Data builder for 1D raking
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    data_builder = DataBuilder(
        dim_specs={'var1': -1},
        value='value',
        weights='weights'
    )

Data builder for 2D raking
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    data_builder = DataBuilder(
        dim_specs={'var1': -1, 'var2': -1},
        value='value',
        weights='weights'
    )

Data builder for 3D raking
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    data_builder = DataBuilder(
        dim_specs={"var1": -1, "var2": -1, "var3": -1},
        value='value',
        weights='weights'
    )

Data builder for USHD raking (level 1 to all-cause)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    data_builder = DataBuilder(
        dim_specs={"cause": -1, "race": -1, "county": -1},
        value="value",
        weights="weights",
    )

Data builder for USHD raking (level 2 to level 1 parent cause)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    data_builder = DataBuilder(
        dim_specs={"cause": -1, "race": -1, "county": -1},
        value="value",
        weights="weights",
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

