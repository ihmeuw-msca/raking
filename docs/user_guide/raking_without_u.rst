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
