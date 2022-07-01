from datetime import timedelta

import mplfinance as mpf
import torch
import torch.nn.functional as F
from src.stock_prediction.components.utils import read_img

edge_color = mpf.make_marketcolors(up="g", down="r", wick="inherit", volume="inherit", edge="inherit", alpha=1.0)
custom_style = mpf.make_mpf_style(base_mpf_style='yahoo', figcolor="black", marketcolors=edge_color)
width_config = {"candle_linewidth": 2,
                "candle_width": 0.87,
                "volume_width": 0.8}


@torch.no_grad()
def predict(stock_df, model, transform, device):
    x_min = stock_df.index[0] - timedelta(days=1)
    x_max = stock_df.index[-1] + timedelta(days=1)

    fig, _ = mpf.plot(stock_df,
                      type="candle",
                      style=custom_style,
                      update_width_config=width_config,
                      figsize=(10, 10),
                      fontscale=0,
                      xlim=(x_min, x_max),
                      axisoff=True,
                      tight_layout=True,
                      returnfig=True,
                      closefig=True)
    fig.savefig("app/temp/image.png", bbox_inches="tight")
    model.eval()
    image = read_img("app/temp/image.png", (224, 224))
    image = transform(image).reshape((1, 3, 224, 224))
    image = image.to(device)
    out = F.softmax(model(image), 1)
    prob, pred = torch.max(out, dim=1)
    pred = pred[0].item()
    prob = prob[0].item()

    return pred, prob
