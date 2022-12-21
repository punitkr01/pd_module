import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import datetime as dt
import pickle
import faiss
from scipy import linalg
import dateutil.tz
import datetime as dt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error

dfData = pd.read_csv('dfData_bb_trade_sizing_mar_22.csv') 
dfData=dfData[dfData['dpd_status_bb']!='loan_closed']

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
st.title("Sizing")

row0_1, row0_2 = st.columns((1,4))#(1,1))
temp_state_list = dfData.state.unique()
select_state =  row0_1.selectbox("Select State: ", options=temp_state_list,index=0)

temp_district_list = dfData[dfData['state']==select_state].district.unique()
select_district =  row0_1.selectbox("Select District: ", options=temp_district_list,index=0)

temp_tier_list = dfData[dfData['district']==select_district].tier.unique()
select_tier =  row0_1.selectbox("Select Tier: ", options=temp_tier_list,index=0)

temp_pincode_list = dfData[(dfData['district']==select_district)&(dfData['tier']==select_tier)].pincode.unique()
select_pincode =  row0_1.selectbox("Select Pincode: ", options=temp_pincode_list,index=0)

temp_member_group_list = dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)].member_group.unique()
select_member =  row0_1.selectbox("Select Member Group: ", options=temp_member_group_list,index=0)

temp_ticket_size_list = dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)&(dfData['member_group']==select_member)].ticket_size_bb.unique()
select_ticket =  row0_1.selectbox("Select Ticket Size: ", options=temp_ticket_size_list,index=0)

temp_score_band_list = dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)&(dfData['member_group']==select_member)&(dfData['ticket_size_bb']==select_ticket)].score_band_bb.unique()
select_score_band =  row0_1.selectbox("Select Score Band: ", options=temp_score_band_list,index=0)

temp=dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)&(dfData['member_group']==select_member)&(dfData['ticket_size_bb']==select_ticket)&(dfData['score_band_bb']==select_score_band)]

temp=temp[['opened','trades','live_closed','sanctioned_amount','current_balance','dpd_status_bb']]
temp=temp[temp['dpd_status_bb']!='loan_closed']

temp=temp[temp['trades']>0]
temp=temp.sort_values(['trades'],ascending=False)
temp['sanctioned_amount_mean']=temp['sanctioned_amount']/temp['trades']

temp_group= temp.groupby(['dpd_status_bb'],as_index=False).agg({'trades':['count','sum'],'sanctioned_amount_mean':'mean'})
temp_group.columns=['dpd_status_bb','count_trades','num_trades','mean_sanctioned_amount']
temp_group['mean_sanctioned_amount']=temp_group['mean_sanctioned_amount'].astype(int)
temp_group['dpd_%_contrib'] = (temp_group['num_trades'] / temp_group['num_trades'].sum()) * 100
temp_group=temp_group[['dpd_status_bb','dpd_%_contrib','count_trades','num_trades','mean_sanctioned_amount']]
temp_group['dpd_%_contrib'] = np.round(temp_group['dpd_%_contrib'], decimals = 3)
temp_group=temp_group.drop(columns=['count_trades'],axis=1)
#temp_group['dpd_%_contrib'].round(decimals =2)
#temp_group['dpd_%_contrib']=temp_group['dpd_%_contrib'].astype(int)
dpd_90_plus =  temp_group[temp_group['dpd_status_bb']=='90+']['dpd_%_contrib'].values[0]
print(dpd_90_plus)
if (dpd_90_plus > 5.0):
    row0_2.write("'90+' DPD > 5% ")
else:
    row0_2.write("'90+' DPD < 5% ")
#st.table(df.style.format({"E": "{:.2f}"}))
row0_2.write(temp_group.style.format({"dpd_%_contrib": "{:.2f}"}))

row0_2.write("Raw Data")
temp=temp.reset_index()
temp=temp.drop(columns=['index'],axis=1)
temp['sanctioned_amount_mean']=temp['sanctioned_amount_mean'].astype(int)

row0_2.write(temp)

                                                       