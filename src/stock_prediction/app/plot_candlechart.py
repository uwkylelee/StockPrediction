import json

import plotly
import plotly.graph_objects as go
import plotly.subplots as ms


def determine_color(open, close, row, stock_df):
    if close > open:
        return "increasing"
    elif close == open:
        if row == 0:
            return "increasing"
        else:
            close_prev = stock_df.iloc[row - 1]["Close"]
            if close >= close_prev:
                return "increasing"
            else:
                return "decreasing"
    else:
        return "decreasing"


def plot_candlechart(stock_df, stock_name, stock_code):
    stock_df["Status"] = stock_df.apply(lambda x: determine_color(x["Open"], x["Close"], x.name, stock_df), axis=1)

    candle = go.Candlestick(x=stock_df["Date"],
                            open=stock_df["Open"],
                            high=stock_df["High"],
                            low=stock_df["Low"],
                            close=stock_df["Close"],
                            name="",
                            increasing_line_color="red",
                            increasing_fillcolor="red",
                            decreasing_line_color="blue",
                            decreasing_fillcolor="blue")

    volume_inc = go.Bar(x=stock_df.loc[stock_df["Status"] == "increasing"]["Date"],
                        y=stock_df.loc[stock_df["Status"] == "increasing"]["Volume"],
                        name="",
                        marker={"color": "red"},
                        xperiodalignment="middle")

    volume_dec = go.Bar(x=stock_df.loc[stock_df["Status"] == "decreasing"]["Date"],
                        y=stock_df.loc[stock_df["Status"] == "decreasing"]["Volume"],
                        name="",
                        marker={"color": "blue"},
                        xperiodalignment="middle")

    fig = ms.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[1200, 300])
    fig.add_trace(candle, row=1, col=1)
    fig.add_trace(volume_inc, row=2, col=1)
    fig.add_trace(volume_dec, row=2, col=1)

    fig.update_layout(
        title=f"{stock_name}: {stock_code}",
        title_x=0.5,
        boxgap=0,
        bargap=0.25,
        yaxis1_title='Price',
        yaxis2_title='Volume',
        xaxis2_title='Date',
        margin={"t": 50,
                "b": 10,
                "l": 0,
                "r": 10},
        showlegend=False,
        xaxis1_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False)

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return plot_json
