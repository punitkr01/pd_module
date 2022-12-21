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


dfData = pd.read_csv('dfData_bb_trade_sizing_mar_22_live_groupby.csv')
dfData=dfData.sort_values(['num_trades'],ascending=False)
#dfData = pd.read_csv('dfData_bb_trade_sizing_mar_22_live.csv') 
#dfData=dfData[dfData['dpd_status_bb']!='loan_closed']

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
st.markdown(
    """
<style>
span[data-baseweb="tag"] {
  background-color: green !important;
}
</style>
""",
    unsafe_allow_html=True,
)
#Reduce Padding on top from default 6 to 1 
st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

#app title
st.subheader("District Profiling")

#st.title("CV Sizing")

row0_1, row0_2 = st.columns((1,4))#(1,1))
temp_state_list = dfData.state.unique()
select_state =  row0_1.selectbox("Select State: ", options=temp_state_list,index=0)

temp_district_list = dfData[dfData['state']==select_state].district.unique()
select_district =  row0_1.selectbox("Select District: ", options=temp_district_list,index=0)

temp_tier_list = dfData[dfData['district']==select_district].tier.unique()
select_tier =  row0_1.multiselect("Select Tier: ", options=temp_tier_list,default=temp_tier_list)#,index=0)

pincode_true = 0

if(pincode_true==1):
    temp_pincode_list = dfData[(dfData['district']==select_district)&(dfData['tier']==select_tier)].pincode.unique()
    select_pincode =  row0_1.multiselect("Select Pincode: ", options=temp_pincode_list,index=0)

    temp_member_group_list = dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)].member_group.unique()
    select_member =  row0_1.multiselect("Select Member Group: ", options=temp_member_group_list,index=0)

    temp_ticket_size_list = dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)&(dfData['member_group']==select_member)].ticket_size_bb.unique()
    select_ticket =  row0_1.multiselect("Select Ticket Size: ", options=temp_ticket_size_list,index=0)

    temp_score_band_list = dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)&(dfData['member_group']==select_member)&(dfData['ticket_size_bb']==select_ticket)].score_band_bb.unique()
    select_score_band =  row0_1.multiselect("Select Score Band: ", options=temp_score_band_list,index=0)

    temp=dfData[(dfData['pincode']==select_pincode)&(dfData['tier']==select_tier)&(dfData['member_group']==select_member)&(dfData['ticket_size_bb']==select_ticket)&(dfData['score_band_bb']==select_score_band)]
else:
    #temp_pincode_list = dfData[(dfData['district']==select_district)&(dfData['tier']==select_tier)].pincode.unique()
    #select_pincode =  row0_1.selectbox("Select Pincode: ", options=temp_pincode_list,index=0)

    temp_member_group_list = dfData[(dfData['district']==select_district)&(dfData['tier'].isin(select_tier))].member_group.unique()
    select_member =  row0_1.multiselect("Select Member Group: ", options=temp_member_group_list,default=temp_member_group_list)

    temp_ticket_size_list = dfData[(dfData['district']==select_district)&(dfData['tier'].isin(select_tier))&(dfData['member_group'].isin(select_member))].ticket_size_bb.unique()
    select_ticket =  row0_1.multiselect("Select Ticket Size: ", options=temp_ticket_size_list,default=temp_ticket_size_list)#index=0)

    temp_score_band_list = dfData[(dfData['district']==select_district)&(dfData['tier'].isin(select_tier))&(dfData['member_group'].isin(select_member))&(dfData['ticket_size_bb'].isin(select_ticket))].score_band_bb.unique()
    temp_band_list = ['band_4','band_3','band_2','band_1','low']
    temp_score_band_list_1 = intersection(temp_band_list,temp_score_band_list)#
    select_score_band =  row0_1.multiselect("Select Score Band: ", options=temp_score_band_list_1,default=temp_score_band_list_1)

    temp=dfData[(dfData['district']==select_district)&(dfData['tier'].isin(select_tier))&(dfData['member_group'].isin(select_member))&(dfData['ticket_size_bb'].isin(select_ticket))&(dfData['score_band_bb'].isin(select_score_band))]

temp = temp.groupby(['dpd_status_bb'],as_index=False).agg({'num_trades':'sum','mean_sanctioned_amount':'mean'})
temp.columns = ['dpd_status_bb','num_trades','mean_sanctioned_amount']
#temp=temp[['num_trades','mean_sanctioned_amount','dpd_status_bb']]
temp=temp.sort_values(['num_trades'],ascending=False)
temp['mean_sanctioned_amount']=temp['mean_sanctioned_amount'].astype(int)
temp['dpd_%_contrib'] = (temp['num_trades'] / temp['num_trades'].sum()) * 100
temp=temp[['dpd_status_bb','dpd_%_contrib','num_trades','mean_sanctioned_amount']]
temp['dpd_%_contrib'] = np.round(temp['dpd_%_contrib'], decimals = 3)
temp['dpd_%_contrib'].round(decimals =2)
temp=temp.reset_index()
temp=temp.drop(columns=['index'],axis=1)
#temp_group['dpd_%_contrib']=temp_group['dpd_%_contrib'].astype(int)

if temp[temp['dpd_status_bb']=='90+'].shape[0]>0:
    dpd_90_plus =  temp[temp['dpd_status_bb']=='90+']['dpd_%_contrib'].values[0]
    print(dpd_90_plus)
    if (dpd_90_plus > 5.0):
        row0_2.write("'90+' DPD > 5% ")
    else:
        row0_2.write("'90+' DPD < 5% ")
else:
    row0_2.write("No 90+ DPD ")
#st.table(df.style.format({"E": "{:.2f}"}))
row0_2.write(temp.style.format({"dpd_%_contrib": "{:.2f}"}))

#row0_2.write("Raw Data")
#temp=temp.reset_index()
#temp=temp.drop(columns=['index'],axis=1)
#temp['sanctioned_amount_mean']=temp['sanctioned_amount_mean'].astype(int)

#row0_2.write(temp)

                                                       