Raking without uncertainty
==========================

In this case, we supposed that we have only point estimates (no draws) for the observations and the margins. We compute the raked values but we do not compute the uncertainty on the raked values as we do not have uncertainty on the observations and the margins.

Input files format
------------------

1D raking
^^^^^^^^^

The observations and the margin are entered as pandas data frames. The observations data frame must have at least one column named value that contains the values of the observations and another column that contains the values of the categorical variable (e.g. name of county if we rake from counties to state value). The margin data frame contains one row and must have at least one column named "value_agg_over_var1" if "var1" is the name of the categorical variable over which we rake (e.g. if the column in the observations data frame is called "county", then the column in the margin data frame would be called "value_agg_over_county").

The observations data frame looks like this:

=====  ======
value  var1
=====  ======
float  string
float  string
float  string
float  string
=====  ======

2D raking
^^^^^^^^^

The observations and the margins are entered as pandas data frames. The observations data frame must have at least one column named value that contains the values of the observations and two other columns that contain the values of the two categorical variables. The first margins data frame must have at least one column named "value_agg_over_var1" if "var1" is the name of the first categorical variable over which we rake, and one column that contains the values of the second categorical variable. The second margins data frame must have at least one column named "value_agg_over_var2" if "var2" is the name of the second categorical variable over which we rake, and one column that contains the values of the first categorical variable.

The observations data frame looks like this:

=====  ====== ======
value  var1   var2
=====  ====== ======
float  string string
float  string string
float  string string
float  string string
=====  ====== ======

The first margins data frame looks like this:

====== ===================
var2   value_agg_over_var1
====== ===================
string float
string float
string float
string float
====== ===================

The second margins data frame looks like this:

====== ===================
var1   value_agg_over_var2
====== ===================
string float
string float
string float
string float
====== ===================

3D raking
^^^^^^^^^

The observations and the margins are entered as pandas data frames. The observations data frame must have at least one column named value that contains the values of the observations and three other columns that contain the values of the three categorical variables. The first margins data frame must have at least one column named "value_agg_over_var1" if "var1" is the name of the first categorical variable over which we rake, and two columns that contains the values of the second and the third categorical variables. The second margins data frame must have at least one column named "value_agg_over_var2" if "var2" is the name of the second categorical variable over which we rake, and two columns that contains the values of the first and the third categorical variables. The third margins data frame must have at least one column named "value_agg_over_var3" if "var3" is the name of the third categorical variable over which we rake, and two columns that contains the values of the first and the second categorical variables.

The observations data frame looks like this:

=====  ====== ====== ======
value  var1   var2   var3
=====  ====== ====== ======
float  string string string
float  string string string
float  string string string
float  string string string
=====  ====== ====== ======

The first margins data frame looks like this:

====== ====== ===================
var2   var3   value_agg_over_var1
====== ====== ===================
string string float
string string float
string string float
string string float
====== ====== ===================

The second margins data frame looks like this:

====== ====== ===================
var1   var3   value_agg_over_var2
====== ====== ===================
string string float
string string float
string string float
string string float
====== ====== ===================

The third margins data frame looks like this:

====== ====== ===================
var1   var2   value_agg_over_var3
====== ====== ===================
string string float
string string float
string string float
string string float
====== ====== ===================

Additional columns
^^^^^^^^^^^^^^^^^^

You may specify weights for the observations to be used by the raking process. These weights could be the standard deviations or the variances of the observations. You simply need to add a column with the weights for each of the observations to the observation data frame. What is actually used is the inverse of the weights. If the weight is small, this observation will be given a high importance in the raking process and its raked value will stay very close to the initial value. If the wight is big, this observation will be given a low importance in the raking process and its raked value can differ quite a lot from its initial value. Observations with null standard deviation (no uncertainty) can thus stay unchanged after the raking process because their weight would be zero.

If you do logit raking, you can add lower and upper boundaries for the raked values. For example, the number of deaths must stay between 0 and the population number for the corresponding race and county. You will need to add a column for the lower boundary for each observation and a column for the upper boundary for each observation in the observations data frame.

Raking function input
---------------------

The main function called to launch the raking is the function run_raking. The mandatory arguments are dim, df_obs, df_margins, var_names and cov_mat. The other arguments are optional and are only used to customize the raking. You may also want to specify weights if you want to add weights to the observations and upper and lower if you do logit raking.

* dim: Enter 1, 2 or 3 if you want to do 1D, 2D or 3D raking.
* df_obs: Enter the pandas data frame containing the observations. It must be formatted as explained above.
* df_margins: Enter a list of pandas data frames containing the margins. There will be 1, 2 or 3 margins data frames, depending on the dimension you have chosen for the raking.
* var_names: Enter a list of strings giving the names of the columns contataining the categorical variables over which you do the raking (e.g. cause, race, county). The number of categorical variables is equal to the dimension of the raking.
* margin_names: Enter None. This is only used for special cases.
* cov_mat: Enter False, if you do not have draws and do not compute the covariance matrix of the raked values. Enter True otherwise.
* method: Enter the raking method that you which to use. Default use the chi square distance ("chi2"). You can also use the entropic distance ("entropic") and a general distance ("general"). If you want to do logit raking, enter "logit".
* alpha: This is a parameter for the general distance.
* weights: Enter the name of the column containing the weights (use it only if you want weights).
* lower: Enter the name of the column contatining the lower boundary for the raked values (use it only with logit raking).
* upper: Enter the name of the column contatining the upper boundary for the raked values (use it only with logit raking).
* rtol: This is a parameter to check whether the margins are consistent with each other. See the documentation of the Python function numpy.allclose for details.
* atol: This is a parameter to check whether the margins are consistent with each other. See the documentation of the Python function numpy.allclose for details.
* gamma0: This parameter is for the Newton's method to solve the raking problem. In most cases, you won't have to change it.
* max_iter: This is the maximum number of iterations for the Newton's method. In most cases, you won't have to change it.

Raking function output
----------------------

The raking function will return a list of 4 variables. In this case, only the first one is used and contains the observations data frame with an additional column "raked_value" that contains the raked values.

Distance functions
------------------

The raking algorithm find the raked values that minimize the distance between the initial observations and the unknown raked values. In the package, you can choose between four distances. We denote math:`y_i` the initial observations and math:`\beta_i` the corresponding raked values. We assume that there are math:`N` observations.


Chi square distance
^^^^^^^^^^^^^^^^^^^

.. math::
    f^{\chi^2} \left( \beta , y \right) = \sum_{i = 1}^N \frac{1}{2 y_i} \left( \beta_i - y_i \right)^2

Entropic distance
^^^^^^^^^^^^^^^^^

.. math::
    f^{\text{entropic}} \left( \beta , y \right) = \sum_{i = 1}^N \beta_i \log \left( \frac{\beta_i}{y_i} \right) - \beta_i + y_i

General distance
^^^^^^^^^^^^^^^^

.. math::
    f^{text{general}} \left( \beta , y \right) = \sum_{i = 1}^N \frac{1}{\alpha} \left[ \frac{y_i}{\alpha + 1} \left( \frac{\beta_i}{y_i} \right) ^{\alpha + 1} - \beta_i + c \left( y_i \right) \right]

Logit distance
^^^^^^^^^^^^^^

.. math::
    f^{\text{logit}} \left( \beta , y \right) = \sum_{i = 1}^N \left( \beta_i - l_i  \right) \log \frac{\beta_i - l_i}{y_i - l_i} + \left( h_i - \beta_i \right) \log \frac{h_i - \beta_i}{h_i - y_i}.

