# Streamlit Test File

from openbb_terminal.sdk import openbb
import pandas as pd
# help(openbb.stocks.dd)
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from scipy import stats
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import streamlit as st
import Bullish_trendline

st.title('Bullish Trendline Graph Generation')
index=st.selectbox("Index",options=["NSE50","BSE"])
if st.button('Generate'):
    with st.spinner("Extracting"):
        Bullish_trendline.fun_getlist(index)










