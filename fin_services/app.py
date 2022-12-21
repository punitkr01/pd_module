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

#import pdf_parsing.aws_textract

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
BACKGROUND_COLOR = 'green'
COLOR = 'black'

#Reduce Padding on top from default 6 to 1 
st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

#app title
st.subheader("DS Underwriting")
df_cibil_data_fo=pd.read_csv("/home/ubuntu/fin_services/cibil_data/cibil_bb_info_pdf_link.csv")
#df_cibil_data_fo['pdf_link']                 
df_gps_idle = pd.read_csv("/home/ubuntu/fin_services/trucks_idle_profiling/gps_idle_ratio.csv")

#select user
from os import listdir
from os.path import isfile, join

cibil_filepath = "/home/ubuntu/fin_services/cibil_data/test/cibil_pdf"
print(cibil_filepath)
#print(listdir(cibil_filepath))

cibil_files_list = [f for f in listdir(cibil_filepath) if isfile(join(cibil_filepath, f))]

print(f"# files in test {len(cibil_files_list)}")
print(cibil_files_list[0])
#print(cibil_files_list[0][0])
#temp_state_list = []
select_user_filepath =  st.selectbox("Select User CIBIL PDF: ", options=cibil_files_list,index=0)
print(select_user_filepath)

#### PDF PARSING
from pdf_parsing.aws_textract import *
from pd_modeling.feature_prep_etl_test_change_pt_dw_1 import *
from pd_modeling.feature_prep_etl_test_model2_dw_1 import *

parsing_pd_error = 0
pd_modeling_error = 0
pd_df=pd.DataFrame()
pd_1 = -1
pd_0 = -1

pd_df_2=pd.DataFrame()
pd_mod2_1 = -1
pd_mod2_0 = -1
pdf_file_path = "/home/ubuntu/fin_services/cibil_data/test/cibil_pdf/" + select_user_filepath 

def get_cibil_score(cibil_info):
    for item in cibil_info:
        if item.isnumeric():
            return item
    return ''

def explode_json_to_rows(df):
#     df['user_identifier'] = df['user_identifier'].tolist()
    list_cols_to_explode = ['address','phone_no','email','account_info','enquiry','cibil_info_with_factors','account_info_new','user_identifier']
    for col in list_cols_to_explode: 
        df[col] = df[col].apply(lambda x:eval(x))
    df['cibil_score'] = df['cibil_info_with_factors'].apply(lambda x:get_cibil_score(x))
    print(f"***1 explode ***")
    print(df.head())
    all_cols =list(df.columns)
    len_col_vals = []
    for col in list_cols_to_explode:
        len_col_vals.append(len(df[col][0]))
    max_val = max(1,max(len_col_vals))
    # final df
    col_dict = {}
    for col in all_cols:
        if col not in list_cols_to_explode:
            temp = []
            temp.append(df[col][0])
            for i in range(max_val-1):
                temp.append('')
            col_dict[col] = temp
        else:
            temp =[]
            for i in range(len(df[col][0])):
                temp.append(df[col][0][i])
            for i in range(max_val-len(df[col][0])):
                temp.append('')
            col_dict[col] = temp
    df = pd.DataFrame(col_dict)
    print('col dict = ',col_dict)
    print(f"*****1 {df.head()}")
    df['ACCOUNT'] = df['account_info'].apply(lambda x:x['ACCOUNT'] if 'ACCOUNT' in x else '')
    df['DATES'] = df['account_info'].apply(lambda x:x['DATES'] if 'DATES' in x else '')
    df['AMOUNTS'] = df['account_info'].apply(lambda x:x['AMOUNTS'] if 'AMOUNTS' in x else '')
    print(f"*****2 {df.head()}")
    acc_list = df['account_info_new'].tolist()
#     print(acc_list)
#     acc_list.sort(key = lambda x: pd.to_datetime(x,format ='%m-%y'),reverse=True)
    final_acc_list= []
    for item in acc_list:
        if len(item)>0:
            keys = list(item.keys())
            k_1 = [pd.to_datetime(k,format='%m-%y') for k in keys]
            values = list(item.values())
            sorted_key_index = np.argsort(k_1)[::-1]
            sorted_dict = {keys[i]: values[i] for i in sorted_key_index}
            final_acc_list.append(sorted_dict)
        else:
            final_acc_list.append(item)
    df['DPD_INFO'] = final_acc_list #df['account_info_new']
    print(f"*****3 {df.head()}")
    
    del_cols = ['account_info','account_info_new','cibil_info_with_factors']
    for col in del_cols:
        del df[col]
    return df

try:
    df_path = extract_text_and_save_file(pdf_file_path)
    print(df_path.shape)
    path = Path(pdf_file_path).expanduser()
    pages = extract_pages(path)
    df = show_ltitem_hierarchy(pages)
    list_dpd_json = get_dpd_values_pdf_miner(df)
    df_path['account_info_new'] = [list_dpd_json]
    print(f'pdf parsing success 1 : {df_path.shape}')

    fname = (pdf_file_path).split('/')[::-1][0]
    fname = fname.replace('PDF','csv')
    print(f" output filename fname: {fname}")
    out_loc = '/home/ubuntu/fin_services/cibil_data/streamlit_parsed/'
    out_l = out_loc+fname
    print(f" output filename location: {out_l}")
    df_path.to_csv(out_l,index=False)
    
    temp_df = pd.read_csv(out_l)
    #df_pdf_explode = explode_json_to_rows(temp_df)
    #print(f'pdf parsing explode success 2 : {df_pdf_explode.shape}')

    parsing_pd_error=0
    try:
        print(f'** pdf parsing success : {df_path.shape}')
        print(f'** {df_path.head()}')
        st.caption("PDF Extraction")
        st.write(df_path.astype(str))
        #### PD MODELING V0 - CHANGE POINT MODEL
        pd_df = compute_pd(out_l)

        print(f'**pd modeling success : {pd_df.shape}')
        print(f'**pdf modeling success output : {pd_df.output_dict}')#[0])
        pd_1 = pd_df['output_dict'][0][1]
        pd_0 = pd_df['output_dict'][0][0]
        pd_modeling_error = 0
        #print(pd_df['output'].values)
        
        pd_df_2 = compute_pd_model2(out_l)
        print(f'**pd modeling success : {pd_df_2.shape}')
        print(f'**pdf modeling success output : {pd_df_2.output_dict}')#[0])
        pd_mod2_1 = pd_df_2['output_dict'][0][1]
        pd_mod2_0 = pd_df_2['output_dict'][0][0]
        pd_modeling_error = 0

    except:
        st.write("PD Modeling : No DPD info ! Data insufficient to predict!!")
        pd_modeling_error = 1
        parsing_pd_error=1
except:
    st.write("No DPD info ! Data insufficient to predict !")
    parsing_pd_error=1

#out_model_loc = '/home/ubuntu/fin_services/cibil_data/streamlit_model_out/'
#out_model_l = out_loc + fname


#### UX
st.caption("Evaluating User across multiple Signals")
#tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "CBIL", "GPS", "PinCode", "BankStatement"])
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["Summary", "CIBIL", "GPS", "FastTag", "LB", "BankStatement", "District"])#, "Data-Summary"])

with tab1:
     st.caption("User Scores Summary")
     if(parsing_pd_error==0 & pd_modeling_error==0):
         st.write('DS Model 1: Aggregate Model - Probability of Default :',"{:.3f}".format(pd_mod2_1))#.style.format("{:.3}"))  
         st.write('DS Model 2: Change Point - Probability of Default :',"{:.3f}".format(pd_1))#.style)#.format("{:.3}"))
         st.write('CIBIL SCORE :',pd_df['cibil_score'].values[0])
         st.write('CIBIL: # open loans :',pd_df['open_loans'].values[0])
         num_open_loans = pd_df['open_loans'].values[0]
         if (num_open_loans >0):   
             st.write('CIBIL: avg dpd in last 3 months :',((pd_df['last_3_months_dpd_all'].values[0])/num_open_loans))#.style.format("{:.2}"))
             st.write('CIBIL: avg dpd in last 6 months :',((pd_df['last_6_months_dpd_all'].values[0])/num_open_loans))#.style.format("{:.2}"))
             st.write('CIBIL: avg dpd in last 12 months :',((pd_df['last_12_months_dpd_all'].values[0])/num_open_loans))#.style.format("{:.2}"))
             st.write('CIBIL: avg dpd in last 36 months :',((pd_df['last_36_months_dpd_all'].values[0])/num_open_loans))#.style.format("{:.2}"))
         else:
             st.write('CIBIL: dpd in last 3 months :',pd_df['last_3_months_dpd_all'].values[0])
             st.write('CIBIL: dpd in last 6 months :',pd_df['last_6_months_dpd_all'].values[0])
             st.write('CIBIL: dpd in last 12 months :',pd_df['last_12_months_dpd_all'].values[0])
             st.write('CIBIL: dpd in last 36 months :',pd_df['last_36_months_dpd_all'].values[0])

with tab2:
     st.caption("CIBIL Modeling")
     if(parsing_pd_error==0 & pd_modeling_error==0):
         st.table(pd_df.T)
         st.table(pd_df_2.T)   

with tab3:
    st.caption("GPS truck profiling: Loaded:Idle")
    df_gps_idle['segment']=df_gps_idle.apply(lambda s: 'good' if s['non_loaded_to_load_days'] <1.1 else ('bad' if s['non_loaded_to_load_days'] < 2.5 else 'ugly'), axis=1)
    st.table(df_gps_idle)#.style.format("{:.2}") )

# with tab8:
#     st.caption("Signal Summary")
#     st.table(df_cibil_data_fo)

with tab4:
     st.caption("WIP")
with tab5:
     st.caption("WIP")
with tab6:
     st.caption("WIP")

with tab7:
    dfData = pd.read_csv('./pincode_profiling/dfData_bb_trade_sizing_mar_22_live_groupby.csv')
    dfData=dfData.sort_values(['num_trades'],ascending=False)
    #dfData = pd.read_csv('dfData_bb_trade_sizing_mar_22_live.csv') 
    #dfData=dfData[dfData['dpd_status_bb']!='loan_closed']

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    #st.title("Pincode Profiling")

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
        select_member =  row0_1.multiselect("Select Member Group: ", options=temp_member_group_list,default=['PEER'])#temp_member_group_list)

        temp_ticket_size_list = dfData[(dfData['district']==select_district)&(dfData['tier'].isin(select_tier))&(dfData['member_group'].isin(select_member))].ticket_size_bb.unique()
        select_ticket =  row0_1.multiselect("Select Ticket Size: ", options=temp_ticket_size_list,default=['lcv_1','lcv_2','mhcv_1','mhcv_2'])#temp_ticket_size_list)#index=0)

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

