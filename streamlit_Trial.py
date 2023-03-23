# Streamlit Test File

import streamlit as st
#import Bullish_trendline

st.title('Bullish Trendline Graph Generation')
index=st.selectbox("Index",options=["NSE50","BSE"])
if st.button('Generate'):
    with st.spinner("Extracting"):
        #Bullish_trendline.fun_getlist(index)
        st.text("WIP")










