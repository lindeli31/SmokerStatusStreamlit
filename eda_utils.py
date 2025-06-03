import pandas as pd
import altair as alt
import numpy  as np 

def plot_corrplot(corr, thr):
    """_summary_

    Args:
        corr (matrix): correlation matrix 
        the (int): threshold choose to visualize pearson correlation coefficient
    """
    base = alt.Chart(corr).encode(
            x='var2:O',
            y='var1:O'
        )
    
    
    ellipses = base.mark_point(
            shape='circle',
            filled=True,
            size=100
        ).encode(
            color=alt.Color('correlation:Q', 
                       scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                       legend=None),
            size=alt.Size('size:Q', 
                     scale=alt.Scale(range=[10, 100]),
                     legend=None)
        ).transform_filter(
            alt.datum.var1 >= alt.datum.var2  
        )
    
    # Adding text
    text = base.mark_text(
            fontSize=10
        ).encode(
            text=alt.condition(
                (alt.datum.correlation >= thr) | (alt.datum.correlation <= -thr),
                alt.Text('correlation:Q', format='.2f'),
                alt.value('')
            ),
            color=alt.value('black')
        ).transform_filter(
        alt.datum.var1 >= alt.datum.var2
        )
    

    chart = (ellipses + text).properties(
            width=500,
            height=500
        ).configure_axis(
            labelFontSize=10
        )
    return chart
def plot_categorical(
    data: pd.DataFrame,
    var: str,
    group_var: str = "smoking",
    margin: int = 1,
    title: str | None = None,
):
    """
    Plot proportional bar charts for two categorical variables.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data set.
    var : str
        Main categorical variable to analyse (mapped to the *x* axis when
        ``margin == 1`` and to colour otherwise).
    group_var : str, default "smoking"
        Grouping variable (mapped to colour when ``margin == 1`` and to the
        *x* axis otherwise).
    margin : {1, 2}, default 1
        * 1  – proportions within each level of **var**  
        * 2  – proportions within each level of **group_var**
    title : str or None, optional
        Custom plot title.  If *None*, a descriptive title is generated.

    Returns
    -------
    chart : alt.Chart
        The Altair chart object (ready for display in Jupyter, Streamlit, etc.).
    prop_df : pandas.DataFrame
        Long-format table with the computed proportions.  Columns are
        ``[var, group_var, "Proportion"]``.

    Examples
    --------
    >>> chart, prop = plot_categoricalno(df, "gender")
    >>> chart  # displays in a Jupyter notebook
    """
    # ----- 1  Input checks ---------------------------------------------------
    if var not in data.columns:
        raise ValueError(f"`{var}` not found in DataFrame")
    if group_var not in data.columns:
        raise ValueError(f"`{group_var}` not found in DataFrame")
    if margin not in (1, 2):
        raise ValueError("`margin` must be 1 or 2")

    # ----- 2  Compute proportions -------------------------------------------
    counts = pd.crosstab(data[var], data[group_var]).astype(float)

    if margin == 1:           # proportions by rows (levels of `var`)
        prop = counts.div(counts.sum(axis=1), axis=0)
    else:                     # proportions by columns (levels of `group_var`)
        prop = counts.div(counts.sum(axis=0), axis=1)

    prop_df = (
        prop.reset_index()
            .melt(id_vars=var, var_name=group_var, value_name="Proportion")
    )

    # ----- 3  Automatic labels ----------------------------------------------
    if title is None:
        if margin == 1:
            title = f"Proportions of {group_var} by level of {var}"
        else:
            title = f"Proportions of {var} by level of {group_var}"

    x_field     = var        if margin == 1 else group_var
    offset_field = group_var if margin == 1 else var
    color_field  = offset_field  # same field for colour and dodge offset

    # ----- 4  Build Altair chart --------------------------------------------
    base = alt.Chart(prop_df).encode(
        x=alt.X(f"{x_field}:N", title=x_field),
        y=alt.Y("Proportion:Q", title="Proportion"),
        color=alt.Color(f"{color_field}:N", legend=alt.Legend(title=color_field)),
        xOffset=f"{offset_field}:N",          # <-- *dodges* the bars
        tooltip=[
            alt.Tooltip(f"{x_field}:N",      title=x_field),
            alt.Tooltip(f"{offset_field}:N", title=offset_field),
            alt.Tooltip("Proportion:Q",      format=".3f")
        ]
    )

    bars = base.mark_bar()
    labels = base.mark_text(dy=-5).encode(
        text=alt.Text("Proportion:Q", format=".3f")
    )

    chart = (bars + labels).properties(
        title=title,
        width=alt.Step(30)  # makes room for dodged bars; tweak as desired
    )

    return chart


def scatter_plot(
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    color_var: str = "smoking",
    jitter: bool = True,
    title: str | None = None,
    *,
    jitter_width: float = 0.01,
    seed: int | None = None,
):
    """
    Scatter-plot two continuous variables, coloured by a categorical group.

    Parameters
    ----------
    data : pandas.DataFrame
        Source data set.
    x_var, y_var : str
        Column names to be mapped to the *x* and *y* axes.
    color_var : str, default "smoking"
        Categorical column used for point colour.
    jitter : bool, default True
        When *True*, a small random offset is added to both axes to
        mitigate over-plotting (similar to `geom_jitter()` in ggplot2).
    title : str or None, optional
        Custom chart title.  If *None*, a descriptive title is generated.
    jitter_width : float, default 0.01
        Jitter magnitude expressed as a fraction of each variable’s range.
        Ignored when `jitter=False`.
    seed : int or None, optional
        Random seed for reproducible jitter.

    Returns
    -------
    chart : alt.Chart
        The Altair chart object (ready for notebooks, Streamlit, etc.).

    Notes
    -----
    * Requires **Altair ≥ 5.0** and **NumPy**.
    * The function performs basic column-existence checks and casts the
      colouring variable to `category` to ensure consistent legends.
    * Jitter is applied in Python before plotting; if you need *exact*
      original values for tooltips, pass `jitter=False`.
    """
    # ---- 1. Input validation ------------------------------------------------
    missing = [c for c in (x_var, y_var, color_var) if c not in data.columns]
    if missing:
        raise ValueError(f"Column(s) not found in DataFrame: {', '.join(missing)}")

    # ---- 2. Make a working copy & ensure categorical colour -----------------
    df = data.copy()
    df[color_var] = df[color_var].astype("category")

    # ---- 3. Apply jitter if requested --------------------------------------
    x_field, y_field = x_var, y_var  # default
    if jitter:
        rng = np.random.default_rng(seed)
        dx = jitter_width * (df[x_var].max() - df[x_var].min())
        dy = jitter_width * (df[y_var].max() - df[y_var].min())
        df["_x_jit"] = df[x_var] + rng.normal(0, dx, size=len(df))
        df["_y_jit"] = df[y_var] + rng.normal(0, dy, size=len(df))
        x_field, y_field = "_x_jit", "_y_jit"

    # ---- 4. Auto-title ------------------------------------------------------
    if title is None:
        title = f"{y_var} vs. {x_var} coloured by {color_var}"

    # ---- 5. Build Altair chart ---------------------------------------------
    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_var),
            y=alt.Y(f"{y_field}:Q", title=y_var),
            color=alt.Color(f"{color_var}:N",
                        scale=alt.Scale(scheme="category10"),  # palette più contrastata
                        legend=alt.Legend(title=color_var)),
            tooltip=[
                alt.Tooltip(f"{x_var}:Q", title=x_var),
                alt.Tooltip(f"{y_var}:Q", title=y_var),
                alt.Tooltip(f"{color_var}:N", title=color_var),
            ],
        )
        .properties(width=400, height=400, title=title)
    )

    return chart
