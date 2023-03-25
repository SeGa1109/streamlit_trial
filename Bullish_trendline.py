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

database_min_length = 10
def fun_acendingtriangle(ticker):
    global candleid
    import datetime
    Previous_Date = datetime.datetime.today() - datetime.timedelta(days=1500)
    timeframe = 1440

    df = openbb.stocks.load(
        symbol=ticker,
        start_date=Previous_Date,
        end_date='',
        interval=timeframe,
        prepost=False,
        source='YahooFinance',
        weekly=False
    )
    if(len(df)<database_min_length):
        return print("no dataaaaaaaaaaaaaa")
    df = df.reset_index()
    df['Ticker'] = ticker
    # df.to_csv('openbb_data.csv', index=True)
#######################################################################################
    def pivotid(df1, l, n1, n2):
        if l - n1 < 0 or l + n2 >= len(df1):
            return 0

        pividhigh = 1
        for i in range(l - n1, l + n2 + 1):
            if (df1.High[l] < df1.High[i]):
                pividhigh = 0
        if pividhigh:
            return 2
        else:
            return 0

    df['pivot'] = df.apply(lambda x: pivotid(df, x.name, 25, 25), axis=1)
    # df.to_csv('pivot.csv', index=True)

###########################################################################################
    import numpy as np
    def pointpos(x):
        if x['pivot'] == 2:
            return x['High'] + 1e-3
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    df.to_csv('pointpos.csv', index=True)

############################################################################## Plot linear reg line

    # initialize arrays
    maxim = np.array([])
    xxmax = np.array([])

    # get the index and high values of rows where pivot is equal to 2
    mask = df['pivot'] == 2
    df_temp = df[mask]
    maxim = np.append(maxim, df_temp['High'].tail(2).values)
    xxmax = np.append(xxmax, df_temp.index[-2:])


    slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)
    prediction = slmax * df.index[-1] + intercmax

    # print(prediction)
    if abs(df.Close.iloc[-1] - prediction) <= 0.04 * prediction:
        #plot graph
        start_index = df.index.get_loc(xxmax[0] - 20)
        dfpl = df[start_index:]
        # print(xxmax[0])
        # create chart
        fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                                             open=dfpl['Open'],
                                             high=dfpl['High'],
                                             low=dfpl['Low'],
                                             close=dfpl['Close'])])
        # add linear regression line
        xxmax_extended = np.array(list(range(int(xxmax[0]), int(df.index[-1]))))
        fig.add_trace(
            go.Scatter(x=xxmax_extended, y=slmax * xxmax_extended + intercmax, mode='lines', name='max slope'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        # Update x-axis to show dates
        dfpl['date'] = dfpl['date'].dt.strftime('%m-%d')
        fig.update_layout(xaxis=dict(
            tickmode='array',
            tickvals=dfpl.index,
            ticktext=dfpl['date']
        ))
        # fig.show()
        fig.write_image(
            "G:\PyGit\pythonProject\generated\{}.jpeg".format(
                ticker))
        image=Image.open("G:\PyGit\pythonProject\generated\{}.jpeg".format(
                ticker))
        return image
        #st.image(image,caption=ticker)
    else:
        print("out of range")
        return None

def fun_getlist(a):
    inputlistpath=r"G:\PyGit\pythonProject\{}.csv".format(a)
    all_us_script_list = pd.read_csv(inputlistpath, header=None)
    all_us_script_list_2 = all_us_script_list[0].tolist()
    i=0
    j=1
    for item2 in all_us_script_list_2[5:]:
        op=fun_acendingtriangle(item2)
        if op == None:
            continue
        else:
            i+=1
            if i ==1:
                locals()['col'+str(j)],locals()['col'+str(j+1)]=st.columns([1,1])
                locals()['col'+str(j)].image(op,caption=item2)
            if i==2:
                locals()['col' + str(j+1)].image(op,caption=item2)
                i=0
                j+=2



# fun_getlist("2_nasdaq script list")#
#fun_getlist("NIFTY500")
# fun_getlist("SP500")
# fun_acendingtriangle('INFY.NS')
