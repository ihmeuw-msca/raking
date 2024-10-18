import altair as alt
import numpy as np
import pandas as pd

# Data
n_counties = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
size = [663, 1323, 1983, 2643, 3303, 3963, 4623, 5283, 5943, 6603]
lu_time = [
    0.14339,
    1.27817,
    4.98063,
    13.0978,
    24.6151,
    41.4776,
    70.0496,
    94.7001,
    132.092,
    178.872,
]
sp_time = [
    0.190271,
    0.501126,
    0.941338,
    1.52781,
    2.17216,
    3.0226,
    4.85216,
    5.25561,
    6.49375,
    8.61088,
]

# Dataframe
df = pd.DataFrame(
    {
        "n_counties": n_counties + n_counties,
        "size": size + size,
        "Method": ["LU"] * len(n_counties) + ["SP"] * len(n_counties),
        "time": lu_time + sp_time,
    }
)

# Divide computation time by 2 (chi2 and entropic were computed)
df["time"] = df["time"] / 2.0

# Plot
chart = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x=alt.X("size:O", axis=alt.Axis(title="Matrix size")),
        y=alt.Y("time:Q", axis=alt.Axis(title="Computation time")).scale(
            type="log"
        ),
        color=alt.Color("Method:N"),
        strokeDash="Method:N",
    )
    .configure_axis(labelFontSize=16, titleFontSize=16)
    .configure_legend(labelFontSize=16, titleFontSize=16)
)
chart.save("profiling_uncertainty.png")
