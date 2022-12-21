import pandas as pd
import numpy as np
import dateutil.tz
import datetime as dt
import json
from io import StringIO
import urllib3
import logging
import sys
from json import dumps
import time
import argparse
import pickle
from datetime import date

curr_date =  date.today()
xgb_model = pickle.load(open('/home/ubuntu/fin_services/pd_modeling/xgb_cibil_classifier_v2.pkl', "rb"))
out_loc='/home/ubuntu/fin_services/cibil_data/model_out/v2/'
def get_cibil_score(cibil_info):
    for item in cibil_info:
        if item.isnumeric():
            return item
    return ''
def get_static_raw_cibil_features(df):
    dfs=[]
    for idx,row in df.iterrows():
        
        user_id = row.name
        gender = row['gender']
        name = row['name']
        total_email = len(row['email'])
        total_phone_nos = len(row['phone_no'])
        dob = row['dob']
        user_identifier = row['user_identifier']
        # remove from startic in final version
#         age = get_age(date(int(dob_list[2]),int(dob_list[1]),int(dob_list[0])))
        cibil_score = get_cibil_score(row['cibil_info_with_factors'])
        total_address = len(row['address'])
        total_loans = len(row['account_info'])

        df_static = pd.DataFrame({ 'user_id' : [user_id], 'name':[name],'gender':[gender], 'total_email':[total_email],'dob':[dob],'cibil_score':[cibil_score],
                       'total_address':[total_address],'total_loans':[total_loans],'total_phone_nos': [total_phone_nos], 'user_identifier':[user_identifier] })
        dfs.append(df_static)
    return pd.concat(dfs)

def get_timestamp(dpd_list):
    final_json = {}
    if (len(dpd_list)) > 0:
        
        yrs = list(dpd_list.keys())[::-1]
        
        for item in yrs:
            dpd_elem = dpd_list[item]
            for k,v in dpd_elem.items():
    #             print(k,v)
                final_json[k+'-'+item]=v
    return final_json

def get_dpd_raw_table(df):
    dfs = [] 
    for idx,row in df.iterrows():
        user_id = row.name 
        acc_info_list = row['account_info']
        dpd_info_json = row['account_info_new']
        for i in range(len(acc_info_list)):
            loan_id = i 
            loan_info = acc_info_list[i]
            sanc_amount =''
            loan_type = loan_info['ACCOUNT']['TYPE']
            ownership = loan_info['ACCOUNT']['ownership']
            if 'sanctioned' in loan_info['AMOUNTS']:
                sanc_amount=loan_info['AMOUNTS']['sanctioned']
            interest_rate = loan_info['AMOUNTS']['interest_rate']
            repayment_tenure = loan_info['AMOUNTS']['repay_tenure']
            emi_amount = loan_info['AMOUNTS']['emi']
            pmt_freq = loan_info['AMOUNTS']['pmt_freq']
            open_date = loan_info['DATES']['opened']
            closed_date = loan_info['DATES']['closed']
#             dpd_list = loan_info['PaymentHistory']['dayPayDue']
            timestamp_json = dpd_info_json[i]
            for k,v in timestamp_json.items():
                is_open=1
                is_closed=0
                ## dd-mm-yyyy
                
                try:

                    curr_timestamp = pd.to_datetime(k,format='%m-%y')
                    closed_date_m_y = pd.to_datetime(closed_date,dayfirst=True)
                    if len(closed_date.split('-'))>=3:
                        
                        if curr_timestamp.year==closed_date_m_y.year and curr_timestamp.month==closed_date_m_y.month:
                            is_open=0
                            is_closed=1
                    df_dpd = pd.DataFrame({'user_id':[user_id],'loan_id':[loan_id],'timestamp':[k],'dpd':[v],'loan_type':[loan_type],'ownership':[ownership],'sanc_amount':[sanc_amount], 'interest_rate':[interest_rate], 
                                  'repayment_tenure':[repayment_tenure],'emi_amount':[emi_amount],'pmt_freq':[pmt_freq],'open_date':[open_date],
                                  'closed_date':[closed_date],'is_open':[is_open],'is_closed':[is_closed]})
                    dfs.append(df_dpd)
                except:
                    
                    break
                

    if len(dfs)<1:
        
        return pd.DataFrame({})
                        
    return pd.concat(dfs)
def post_process_raw_dpd(df):
#     df['next_loan_id'] = df['loan_id'].shift(-1)

    df['next_user_id'] = df['user_id'].shift(-1)
#     df['next_loan_id'] = df['next_loan_id'].fillna(0)
    df['next_user_id'] = df['next_user_id'].fillna(0)
#     df['next_loan_id'] = df['next_loan_id'].apply(lambda x: int(x)  )
    df['next_user_id'] = df['next_user_id'].apply(lambda x: int(x) )

    return df

# def open_closed_status(row):
#     if row['closed_date'] !='':
#         if row['timestamp']==row['closed_date'][3:]:
#             row['is_open']=0
#             row['is_closed']=1
#     return row
def get_sanc_amt(sanc_string):
    if sanc_string=='' or sanc_string is None:
        return 0
    sanc_elems =sanc_string.split(',')
    final_amt=''
    for item in sanc_elems:
        final_amt+=item
    return eval(final_amt)   

def get_enquiry_table(df): 
    df_n = []
    #print(df.enquiry)
    for idx,row in df.iterrows():
        user_id = row.name
        enq_info = row['enquiry']
        for elem in enq_info:
            date = elem['enquiry_date']
            enq_purpose = elem['enquiry_purpose']
            try:
                enq_amt = get_sanc_amt(elem['enquiry_amount'])
            except:
                enq_amt=0
            df_enq = pd.DataFrame({'user_id':[user_id],'date':[date],'enq_purpose':[enq_purpose],'enq_amount':[enq_amt]})
            df_n.append(df_enq)
         
    if len(df_n)<1:
        return pd.DataFrame({'user_id':[user_id],'date':[''],'enq_purpose':[''],'enq_amount':['']})
    return pd.concat(df_n)
def get_flags(acc_info):
    for acc in acc_info:
        dpd_list = acc['PaymentHistory']['dayPayDue']
        timestamp_json = get_timestamp(dpd_list)
        for k,v in timestamp_json.items():
            if v=='900':
                return True
    return False

def fill_zero_val_from_loan_type(sanc_amount,loan_type):
    sanc_list =sanc_amt_loan_type_mean_df[sanc_amt_loan_type_mean_df['loan_type']==loan_type]['sanc_amount_temp'].tolist()
    if len(sanc_list)>0:  
        sanc_amt_loan_type = sanc_list[0]
        if int(sanc_amount)==0:
            return sanc_amt_loan_type
    return sanc_amount
 
## use this for feature only 
def bucket_dpd(dpd_val):
    final_val=dpd_val
    if dpd_val=='STD' or dpd_val=='XXX':
        final_val=0  
    if dpd_val.isnumeric():
        dpd_val=int(dpd_val)
        if dpd_val<10:
            final_val=0
        elif dpd_val>=10 and dpd_val<20:
            final_val=1
        elif dpd_val>=20 and dpd_val<30:
            final_val=2
        elif dpd_val>=30 and dpd_val<40:
            final_val=3
        elif dpd_val>=40 and dpd_val<50:
            final_val=4
        elif dpd_val>=50 and dpd_val<60:
            final_val=5
        elif dpd_val>=60 and dpd_val<70:
            final_val=6
        elif dpd_val>=70 and dpd_val<80:
            final_val=7
        elif dpd_val>=80 and dpd_val<90:
            final_val=8
        else:
            final_val=9
    else:
        final_val=0
    return final_val
    
        
def modified_dpd(dpd_val):
    final_val=dpd_val
    if dpd_val=='STD' or dpd_val=='XXX':
        final_val=0  
    if dpd_val.isnumeric():
        dpd_val=int(dpd_val)
    else:
        final_val=0
    return int(final_val)
from datetime import date
 
def get_age(birthdate,curr_date):
    today = date.today()
    age = curr_date.year - birthdate.year - ((curr_date.month, curr_date.day) < (birthdate.month, birthdate.day))
    return age

## steps to eval all fileds : 
## make raw_dpd as central table as this contains all the info regarding user_id and loan_id
## user_id , key --> get from the raw_dpd_table
## for this pair eval all the fields  | for static directly query from static_raw_df | for enq perform all ops from enq_table 
## eval timestamp based on mm-yyyy from the dpd table to eval the dpd related features 
## static -> 'gender', 'total_email', 'dob', 'age', 'cibil_score',
##       'total_address', 'total_loans', 'total_phone_nos'
from datetime import date
feat_dict = {'Gold Loan' : 'gl', 
             'Personal Loan' : 'personal',
             'Commercial Vehicle Loan' : 'cvl',
             'Credit Card' : 'cc'
            }
def get_all_loan_type_feats(df,loan_type):
    new_df = df[df['loan_type']==loan_type]
    max_date = df_raw_dpd['timestamp_new'].max()
    user_id = df_raw_dpd['user_id'].tolist()[0]
    curr_date =  date.today()

    timestamp_new = max_date
    #print((new_df.dpd.tolist()))
    try:
        
        last_3_months_dpd= sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <timestamp_new and x>=(timestamp_new-pd.DateOffset(months=3))))].dpd.tolist())

        #last_3_months_dpd =  sum(new_df[(new_df['timestamp_new'].apply(lambda x: x in (pd.date_range(prev_day, periods=95, freq="-1D"))))]['modified_dpd'].tolist())
        last_6_months_dpd = sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <timestamp_new and x>=(timestamp_new-pd.DateOffset(months=6))))].dpd.tolist())
        last_12_months_dpd = sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <timestamp_new and x>=(timestamp_new-pd.DateOffset(months=12))))].dpd.tolist())
        last_36_months_dpd = sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <timestamp_new and x>=(timestamp_new-pd.DateOffset(months=36))))].dpd.tolist())
    except:
        last_3_months_dpd=last_6_months_dpd=last_12_months_dpd=last_36_months_dpd=0
    df_l = pd.DataFrame({'user_id':[user_id], 'datetime_formatted':[max_date], 'last_3_months_dpd' :[last_3_months_dpd],'last_6_months_dpd' : [last_6_months_dpd],'last_12_months_dpd':[last_12_months_dpd],'last_36_months_dpd':[last_36_months_dpd]
                        })
#     df_l= df[df['loan_type']==loan_type] #'Gold Loan'    
    final_cols = ['user_id','datetime_formatted']
    col_list = ['last_3_months_dpd', 'last_6_months_dpd', 'last_12_months_dpd',
       'last_36_months_dpd']
    df_grp = df_l[['user_id','datetime_formatted', 'last_3_months_dpd', 'last_6_months_dpd', 'last_12_months_dpd',
       'last_36_months_dpd']]
    key_to_add = feat_dict[loan_type]
    for col in col_list:
        temp = col + '_' +key_to_add
        final_cols.append(temp)
    df_grp.columns = final_cols
    
    return df_grp
def get_cibil_feature_table(df_raw_static,df_raw_dpd,df_raw_enquiry):
    count =0 
    df_raw_dpd.reset_index(inplace=True)
    max_date = df_raw_dpd['timestamp_new'].max()
    user_id = df_raw_dpd['user_id'].tolist()[0]
    user_dpd_df = df_raw_dpd[df_raw_dpd['user_id']==user_id]
#         all_loans = list(set(user_dpd_df['loan_id'].tolist()))

    key =str(user_id)
    new_df = df_raw_dpd
    if new_df.shape[0]>1:
        count+=1

        time_list = np.sort(list(set(new_df.sort_values('timestamp_new',ascending=False)['timestamp_new'].tolist())))[::-1]
        #print(time_list)
        timestamp = time_list[0]
        timestamp_new = time_list[1]
        print(f'####**** {timestamp}, {timestamp_new}')

        total_email = df_raw_static[df_raw_static['user_id']==user_id]['total_email'].tolist()[0]
        gender = df_raw_static[df_raw_static['user_id']==user_id]['gender'].tolist()[0]
        dob = df_raw_static[df_raw_static['user_id']==user_id]['dob'].tolist()[0]
        age = get_age(pd.to_datetime(dob),curr_date)
    #         age = df_raw_static[df_raw_static['user_id']==row['user_id']]['age'].tolist()[0]
        cibil_score = df_raw_static[df_raw_static['user_id']==user_id]['cibil_score'].tolist()[0]
        total_address = df_raw_static[df_raw_static['user_id']==user_id]['total_address'].tolist()[0]
    #         total_loans = df_raw_static[df_raw_static['user_id']==row['user_id']]['total_loans'].tolist()[0]
        total_phone_nos = df_raw_static[df_raw_static['user_id']==user_id]['total_phone_nos'].tolist()[0]

        total_loans = len(set(new_df[(new_df['timestamp_new']<=timestamp_new)
                               &(new_df['is_open']==1)]['loan_id'].tolist()))
        closed_loans = len(set(new_df[(new_df['timestamp_new']<=timestamp_new)
                               &(new_df['is_closed']==1)]['loan_id'].tolist()))

        open_loans = total_loans - closed_loans
        dpd = new_df[new_df['timestamp_new']==timestamp]['modified_dpd'].tolist()[0]
#         dpd_bucket = new_df['dpd_bucket'].mean()
#         dpd_provided = row['dpd']
        sanc_amount = 300000
        loan_type = new_df[new_df['timestamp_new']==timestamp_new]['loan_type'].tolist()[0]
        ownership = new_df[new_df['timestamp_new']==timestamp_new]['ownership'].tolist()[0]
    
        
        ## mm-yyyy 
        print(f'**** {timestamp_new}, {timestamp_new-pd.DateOffset(months=3)}')
        last_3_months_dpd= sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <=timestamp_new and x>=(timestamp_new-pd.DateOffset(months=3))))]['modified_dpd'].tolist())

        #last_3_months_dpd =  sum(new_df[(new_df['timestamp_new'].apply(lambda x: x in (pd.date_range(prev_day, periods=95, freq="-1D"))))]['modified_dpd'].tolist())
        last_6_months_dpd = sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <=timestamp_new and x>=(timestamp_new-pd.DateOffset(months=6))))]['modified_dpd'].tolist())
        last_12_months_dpd = sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <=timestamp_new and x>=(timestamp_new-pd.DateOffset(months=12))))]['modified_dpd'].tolist())
        last_36_months_dpd = sum(new_df[(new_df['timestamp_new'].apply(lambda x: x <=timestamp_new and x>=(timestamp_new-pd.DateOffset(months=36))))]['modified_dpd'].tolist())

                ## enquiry level features -> total_enq_till_date  | unique_enquiry_purpose  | total_enq_amt 
        total_enq_till_date = df_raw_enquiry[(df_raw_enquiry['user_id']==user_id) & (df_raw_enquiry['timestamp']<=timestamp_new) & (df_raw_enquiry['timestamp']>timestamp_new-pd.DateOffset(months=3))].shape[0]
#                 unique_enquiry_purpose = df_raw_enquiry[(df_raw_enquiry['user_id']==row['user_id']) & (df_raw_enquiry['timestamp']<timestamp_new)]['enq_purpose'].nunique()
#                 total_enq_amt = df_raw_enquiry[(df_raw_enquiry['user_id']==row['user_id']) & (df_raw_enquiry['timestamp']<timestamp_new)]['enq_amount'].sum()
        final_json={
            'key':[key] , 'timestamp':[timestamp] ,'datetime_formatted' : [timestamp_new], 'cibil_score' : [cibil_score], 
            'total_email' : [total_email], 'gender' : [gender], 'age' : [age] ,'open_loans':[open_loans],'closed_loans':[closed_loans] ,  'total_address' : [total_address],
            'sanc_amount':[sanc_amount],'total_loans' : [total_loans], 'total_phone_nos' : [total_phone_nos], 'dpd':[dpd],
            'last_3_months_dpd' :[last_3_months_dpd],'last_6_months_dpd' : [last_6_months_dpd],'last_12_months_dpd':[last_12_months_dpd],'last_36_months_dpd':[last_36_months_dpd],
             'last_3_months_dpd_all' :[last_3_months_dpd],'last_6_months_dpd_all' : [last_6_months_dpd],'last_12_months_dpd_all':[last_12_months_dpd],'last_36_months_dpd_all':[last_36_months_dpd],
            
#                     'next_3_months_dpd' : [next_3_months_dpd],'next_6_months_dpd':[next_6_months_dpd] , 'next_12_months_dpd':[next_12_months_dpd], 'next_36_months_dpd':[next_36_months_dpd],
            'total_enq_till_date':[total_enq_till_date],'loan_type':[loan_type],'ownership':[ownership]
#                     ,'unique_enquiry_purpose':[unique_enquiry_purpose],'total_enq_amt':[total_enq_amt]

        }
        col_dict = {'last_3_months_dpd':3, 'last_6_months_dpd':6, 'last_12_months_dpd':12,
   'last_36_months_dpd':36}
        for loan_type in list(feat_dict.keys()):
            key_to_add = feat_dict[loan_type]
            for col in list(col_dict.keys()):
                temp = col + '_' +key_to_add

                final_json[temp] = sum(new_df[(new_df['loan_type']==loan_type)&(new_df['timestamp_new'].apply(lambda x: x <=timestamp_new and x>=(timestamp_new-pd.DateOffset(months=col_dict[col]))))]['modified_dpd'].tolist())

        df_feature = pd.DataFrame(final_json)
        return df_feature
    else:
        return pd.DataFrame({})
    return pd.DataFrame({})
        

def get_loan_type_encoder(loan_type):
    if loan_type=='Commercial Vehicle Loan':
        return 5
    elif loan_type=='Consumer Loan':
        return 4
    elif loan_type=='Gold Loan':
        return 3
    elif loan_type=='Personal Loan':
        return 2
    elif loan_type=='Credit Card':
        return 1
    else:
        return 0
def get_ownership_encoder(ownership):
    if ownership=='Individual':
        return 3
    elif ownership=='Guarantor':
        return 2
    else:
        return 1

def pre_process_features(df_cibil_feat):
    df_cibil_feat['datetime_formatted'] = pd.to_datetime(df_cibil_feat.datetime_formatted, errors = 'coerce')


    df_cibil_feat['loan_type'] = df_cibil_feat['loan_type'].apply(get_loan_type_encoder)
    df_cibil_feat['ownership'] = df_cibil_feat['ownership'].apply(get_ownership_encoder)
    df_cibil_feat['gender'] = df_cibil_feat['gender'].apply(lambda x: 1 if x=='Male' else 0)
#     df_cibil_feat['enquiry_purpose'] = df_cibil_feat['enquiry_purpose'].apply(get_enquiry_purpose_encoder)
    return df_cibil_feat

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import glob

def compute_pd_model2(pd_filename):
    dfs = pd.read_csv(pd_filename)
    print(f'dfs.shape = {dfs.shape}')
        
    #print(dfs)
    change_cols = ['cibil_info_with_factors', 'address', 'phone_no', 'email', 'acc_summary', 'account_info','enquiry','account_info_new']
    for col in change_cols:
        dfs[col] = dfs[col].apply(lambda x:eval(x))

    df_static = get_static_raw_cibil_features(dfs)
    df_raw_dpd = get_dpd_raw_table(dfs)
    if df_raw_dpd.shape[0]>0:

        df_raw_dpd['timestamp_new'] = df_raw_dpd['timestamp'].apply(lambda x:pd.to_datetime(x,format='%m-%y'))
        df_raw_dpd_new = post_process_raw_dpd(df_raw_dpd)
        df_raw_dpd_new.sort_values(['user_id','timestamp_new'],inplace=True)
        df_enquiry = get_enquiry_table(dfs)
    #     sanc_amt_loan_type_mean_df = df_raw_dpd_new.groupby('loan_type').agg({'sanc_amount_temp':'mean'}).reset_index()
    #     df_raw_dpd_new['sanc_amount_temp'] =  df_raw_dpd_new['sanc_amount'].apply(lambda x: get_sanc_amt(x))
    #     df_enquiry = get_enquiry_table(dfs)
    #     df_enquiry['timestamp'] = df_enquiry['date'].apply(lambda x: pd.to_datetime(x))
    #     df_raw_dpd_new['sanc_amt_final'] = df_raw_dpd_new.apply(lambda x:fill_zero_val_from_loan_type(x['sanc_amount_temp'],x['loan_type']),axis=1)

        df_raw_dpd_new['modified_dpd'] = df_raw_dpd_new['dpd'].apply(modified_dpd)
        df_raw_dpd_new['dpd'] = df_raw_dpd_new['dpd'].apply(modified_dpd)
    #     df_raw_dpd_new['dpd_bucket'] = df_raw_dpd_new['dpd'].apply(bucket_dpd)


        df_enquiry['timestamp'] = df_enquiry['date'].apply(lambda x: pd.to_datetime(x))
        df_cibil_feature =  get_cibil_feature_table(df_static,df_raw_dpd_new,df_enquiry)
        if df_cibil_feature.shape[0]>0:

            df_cibil_feature.to_csv('temp_feat.csv',index=False)
            df_cibil_feature = pd.read_csv('temp_feat.csv')

            print('Feature prep Done !') 

            print('processing and generating all feats !')

            df_cibil_feature['loan_type'] = 'Commercial Vehicle Loan'
            df_cibil_feature['ownership'] = 'Individual'
        #     df_cibil_feature = df_cibil_feature.apply(get_enquiry_type,axis=1)

            obj_cols = ['last_3_months_dpd', 'last_6_months_dpd', 'last_12_months_dpd',
               'last_36_months_dpd']
            for col in obj_cols:
                df_cibil_feature[col] = df_cibil_feature[col].apply(lambda x:float(x) if x is not None else 0)

            features = pre_process_features(df_cibil_feature)

            X = features[['total_email',
   'gender', 'age', 'open_loans', 'closed_loans', 'total_address','sanc_amount',
    'total_phone_nos', 'total_enq_till_date', 'loan_type', 'ownership',
#        'last_3_months_dpd_gl',
#        'last_6_months_dpd_gl', 'last_12_months_dpd_gl',
#        'last_36_months_dpd_gl',
    'last_3_months_dpd_personal', 'last_6_months_dpd_personal',
   'last_12_months_dpd_personal', 'last_36_months_dpd_personal',
    'last_3_months_dpd_cvl', 'last_6_months_dpd_cvl',
   'last_12_months_dpd_cvl', 'last_36_months_dpd_cvl',
#         'last_3_months_dpd_cc', 'last_6_months_dpd_cc', 'last_12_months_dpd_cc',
#        'last_36_months_dpd_cc',
       'last_3_months_dpd_all', 'last_6_months_dpd_all',
   'last_12_months_dpd_all','last_36_months_dpd_all']]
        #     print(X.dtypes)
            predictions_list = xgb_model.predict_proba(X)[0]
            #fname = (args.doc).split('/')[::-1][0]
            #out_l = out_loc+fname

            class_list = xgb_model.classes_
            out_dict = {}
            for i in range(len(class_list)):
                out_dict[class_list[i]] = predictions_list[i]

            print(out_dict)

            df_x = pd.DataFrame(X)
            print(df_x.shape)
            print(df_x.head())
            df_x['output_dict'] = [out_dict]
            df_x['cibil_score'] = features['cibil_score']

            #dfs['output_dict'] = [out_dict]
            #dfs['output'] = xgb_model.predict(X)
            #dfs.to_csv(out_l,index=False)
            #print('Saved File ',out_l)
            return df_x
        else:
            print('Insufficient DPD info !')


    #     df_cibil_feature.to_csv('all_feat_cibil_pdf_v1.csv',index=False)
    #     df_cibil_feature = pd.read_csv('all_feat_cibil_pdf_v1.csv')
    #     df_cibil_feat = pd.read_csv('all_feat_cibil_pdf_v1.csv')

   
if __name__ == "__main__":
    # Document
    
    parser = argparse.ArgumentParser(description ='Enter docname')
    
    # Adding Arguments
    parser.add_argument('doc',type = str)
    args = parser.parse_args()
    dfs = pd.read_csv(args.doc)
        
    print(dfs)
    change_cols = ['cibil_info_with_factors', 'address', 'phone_no', 'email', 'acc_summary', 'account_info','enquiry','account_info_new']
    for col in change_cols:
        dfs[col] = dfs[col].apply(lambda x:eval(x))

    df_static = get_static_raw_cibil_features(dfs)
    df_raw_dpd = get_dpd_raw_table(dfs)
    if df_raw_dpd.shape[0]>0:

        df_raw_dpd['timestamp_new'] = df_raw_dpd['timestamp'].apply(lambda x:pd.to_datetime(x,format='%m-%y'))
        df_raw_dpd_new = post_process_raw_dpd(df_raw_dpd)
        df_raw_dpd_new.sort_values(['user_id','timestamp_new'],inplace=True)
        df_enquiry = get_enquiry_table(dfs)
    #     sanc_amt_loan_type_mean_df = df_raw_dpd_new.groupby('loan_type').agg({'sanc_amount_temp':'mean'}).reset_index()
    #     df_raw_dpd_new['sanc_amount_temp'] =  df_raw_dpd_new['sanc_amount'].apply(lambda x: get_sanc_amt(x))
    #     df_enquiry = get_enquiry_table(dfs)
    #     df_enquiry['timestamp'] = df_enquiry['date'].apply(lambda x: pd.to_datetime(x))
    #     df_raw_dpd_new['sanc_amt_final'] = df_raw_dpd_new.apply(lambda x:fill_zero_val_from_loan_type(x['sanc_amount_temp'],x['loan_type']),axis=1)

        df_raw_dpd_new['modified_dpd'] = df_raw_dpd_new['dpd'].apply(modified_dpd)
        df_raw_dpd_new['dpd'] = df_raw_dpd_new['dpd'].apply(modified_dpd)
    #     df_raw_dpd_new['dpd_bucket'] = df_raw_dpd_new['dpd'].apply(bucket_dpd)


        df_enquiry['timestamp'] = df_enquiry['date'].apply(lambda x: pd.to_datetime(x))
        df_cibil_feature =  get_cibil_feature_table(df_static,df_raw_dpd_new,df_enquiry)
        if df_cibil_feature.shape[0]>0:

            df_cibil_feature.to_csv('temp_feat.csv',index=False)
            df_cibil_feature = pd.read_csv('temp_feat.csv')

            print('Feature prep Done !') 

            print('processing and generating all feats !')

            df_cibil_feature['loan_type'] = 'Commercial Vehicle Loan'
            df_cibil_feature['ownership'] = 'Individual'
        #     df_cibil_feature = df_cibil_feature.apply(get_enquiry_type,axis=1)

            obj_cols = ['last_3_months_dpd', 'last_6_months_dpd', 'last_12_months_dpd',
               'last_36_months_dpd']
            for col in obj_cols:
                df_cibil_feature[col] = df_cibil_feature[col].apply(lambda x:float(x) if x is not None else 0)

            features = pre_process_features(df_cibil_feature)

            X = features[['total_email',
   'gender', 'age', 'open_loans', 'closed_loans', 'total_address','sanc_amount',
    'total_phone_nos', 'total_enq_till_date', 'loan_type', 'ownership',
#        'last_3_months_dpd_gl',
#        'last_6_months_dpd_gl', 'last_12_months_dpd_gl',
#        'last_36_months_dpd_gl',
    'last_3_months_dpd_personal', 'last_6_months_dpd_personal',
   'last_12_months_dpd_personal', 'last_36_months_dpd_personal',
    'last_3_months_dpd_cvl', 'last_6_months_dpd_cvl',
   'last_12_months_dpd_cvl', 'last_36_months_dpd_cvl',
#         'last_3_months_dpd_cc', 'last_6_months_dpd_cc', 'last_12_months_dpd_cc',
#        'last_36_months_dpd_cc',
       'last_3_months_dpd_all', 'last_6_months_dpd_all',
   'last_12_months_dpd_all','last_36_months_dpd_all']]
        #     print(X.dtypes)
        
            predictions_list = xgb_model.predict_proba(X)[0]
            fname = (args.doc).split('/')[::-1][0]
            out_l = out_loc+fname

            class_list = xgb_model.classes_
            out_dict = {}
            for i in range(len(class_list)):
                out_dict[class_list[i]] = predictions_list[i]

            print(out_dict)
            df_x = pd.DataFrame(X)
            print(df_x.shape)
            print(df_x.head())
            df_x['output_dict'] = [out_dict]
            df_x['cibil_score'] = features['cibil_score']

            #dfs['output_dict'] = [out_dict]
            #dfs['output'] = xgb_model.predict(X)
            #dfs.to_csv(out_l,index=False)
            print('Saved File ',out_l)
            #return df_x
        else:
            print('Insufficient DPD info !')


    #     df_cibil_feature.to_csv('all_feat_cibil_pdf_v1.csv',index=False)
    #     df_cibil_feature = pd.read_csv('all_feat_cibil_pdf_v1.csv')
    #     df_cibil_feat = pd.read_csv('all_feat_cibil_pdf_v1.csv')



