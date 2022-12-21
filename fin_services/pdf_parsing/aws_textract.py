import pandas as pd
import numpy as np
import glob

import json

from io import StringIO
import urllib3
import logging
import sys
from json import dumps
import time

def get_ymd(datetime):
    year = datetime.year
    month = datetime.month
    day = datetime.day
            
    if month < 10:
        month = '0' + str(month)
    if day < 10:
        day = '0' + str(day)
    return year, month, day

def first_day_next_month(date):
    return (date.replace(day=1) + dt.timedelta(days=32)).replace(day=1)

def last_second_of_month(date: str) -> str:
    return str((pd.event_timestamp(date) + pd.offsets.MonthEnd(0)).date()) + " 23:59:59"

def first_second_of_month(date: str) -> str:
    return str((pd.event_timestamp(date) + pd.offsets.MonthBegin(0)).date()) + " 00:00:00"

streamer = StringIO()

def setup_logging():
    logger = logging.getLogger()
    for h in logger.handlers:
        logger.removeHandler(h)
     
    h = logging.StreamHandler(stream = streamer)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                              "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger

def query_log(query_id, table, logger):
    status = wr.athena.get_query_execution(query_id)['Status']['State']
    if wr.athena.get_query_execution(query_id)['Status']['State'] in ['FAILED', 'CANCELLED']:
        logger.critical(table + ': query is in ' + status + ' State. ' + 'QueryID: ' + query_id)
    else:
        logger.info(table + ': query is in ' + status + ' State. ' + 'QueryID: ' + query_id)
    return None


def query_progress(query_id, run_async, table_name):
    if not run_async:
            status = wr.athena.get_query_execution(query_id)['Status']['State']
            while status not in ('SUCCEEDED'):
                if status in ['RUNNING', 'QUEUED']:
                    status = wr.athena.get_query_execution(query_id)['Status']['State']
                elif status == 'FAILED':
                    print('Query Failed')
                    break
                elif status == 'CANCELLED':
                    print('Query Cancelled')
                    break
    else:
        status = wr.athena.get_query_execution(query_id)['Status']['State']
        while status not in ('RUNNING'):
            if status == 'QUEUED':
                time.sleep(2)
                status = wr.athena.get_query_execution(query_id)['Status']['State']
            elif status == 'SUCCEEDED':
                print('Query Succeeded')
                break
            elif status == 'FAILED':
                print('Query Failed')
                break
            elif status == 'CANCELLED':
                print('Query Cancelled')
                break

    query_log(query_id, table_name, logger)
    return status

import boto3
# from trp import Document
import time

# Curent AWS Region. Use this to choose corresponding S3 bucket with sample content


import boto3
import time


def start_job(client, s3_bucket_name, object_name):
    response = None
    response = client.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': object_name
            }})

    return response["JobId"]


def is_job_complete(client, job_id):
    time.sleep(1)
    response = client.get_document_text_detection(JobId=job_id)
    status = response["JobStatus"]
    print("Job status: {}".format(status))

    while(status == "IN_PROGRESS"):
        time.sleep(1)
        response = client.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

    return status


def get_job_results(client, job_id):
    pages = []
    time.sleep(1)
    response = client.get_document_text_detection(JobId=job_id)
    pages.append(response)
    print("Resultset page received: {}".format(len(pages)))
    next_token = None
    if 'NextToken' in response:
        next_token = response['NextToken']

    while next_token:
        time.sleep(1)
        response = client.\
            get_document_text_detection(JobId=job_id, NextToken=next_token)
        pages.append(response)
        print("Resultset page received: {}".format(len(pages)))
        next_token = None
        if 'NextToken' in response:
            next_token = response['NextToken']

    return pages

def aws_textract_processor(doc_name):
    
    mySession = boto3.session.Session()
    awsRegion = mySession.region_name

    s3 = boto3.client('s3')
    with open(doc_name, "rb") as f:
        s3.upload_fileobj(f, "temp-text-extraction",doc_name)

    s3BucketName = "temp-text-extraction" 

    # Amazon S3 client
    s3 = boto3.client('s3')

    # Amazon Textract client
    textract = boto3.client('textract')
    region = "ap-south-1"
    client = boto3.client('textract', region_name=region)

    job_id = start_job(client, s3BucketName, doc_name)
    print("Started job with id: {}".format(job_id))
    if is_job_complete(client, job_id):
        response = get_job_results(client, job_id)
    out_list = []
    for result_page in response:
        for item in result_page["Blocks"]:
            if item["BlockType"] == "LINE":
                out_list.append( item["Text"] )
    return out_list
def dpd_decorator(dpd_list):
    t_end = time.time() + 2
#     print('dpd list =',dpd_list)
    len_dpd =len(dpd_list)
    updated_list=[]
    for item in dpd_list:
        hidden_arr=item.split(' ')
        for elem in hidden_arr:
            updated_list.append(elem)
    dpd_list = updated_list
    new_dict={}
    curr_month_idx=0
    month_arr=[]
    tot_mapped=0
    start_idx=0
    flag=True
    while(len(new_dict)<len_dpd/2 and flag):
        start_idx = curr_month_idx
        for i in range(start_idx,len_dpd):
            if len(dpd_list[i])>=4:
                month_arr.append(dpd_list[i])
                curr_month_idx=i
                tot_mapped+=2
            else:
                curr_month_idx = curr_month_idx+len(month_arr)+1
#                 print(curr_month_idx)
                break
#         print(month_arr)
#         break
        day_dpd = len(month_arr)
        for i in range(len(month_arr)):
            if start_idx+day_dpd<len(dpd_list):
                new_dict[month_arr[i]]=dpd_list[start_idx+day_dpd]
                day_dpd+=1
            else:
                flag=False
#         print(new_dict,len(new_dict))
        if time.time() > t_end:
            break
        month_arr=[]
    return new_dict
def get_processed_df(txt_list):
    
    ## DPD info df 
    dpd_elements = ['MEMBER NAME:',  'OPENED:', 'SANCTIONED:', 'ACCOUNT NUMBER:', 'TYPE:', 'LAST PAYMENT:',  'CURRENT BALANCE:',  'CLOSED:', 'OVER DUE:', 'REPORTED',  'EMI', 'OWNERSHIP',  'PMT HIST START', 'PMT FREQ:', 'PMT HIST END:',  'REPAYMENT TENURE:',  'INTEREST RATE:', 'DAYS PAST DUE/ASSET CLASSIFICATION (UP TO 36 MONTHS; LEFT TO RIGHT)']
    amt_idx_list=[]
    ir_list=[]
    for i in range(len(txt_list)):
        if txt_list[i].find('AMOUNTS') !=-1:
            amt_idx_list.append(i)
    for i in range(len(txt_list)):
        if txt_list[i].find('INTEREST RATE:') != -1:
            ir_list.append(i)
#     mem_name=open_date=last_pay=closed_date=reported_date=loan_type=pmt_hist_start=pmt_hist_end=ownership=dpd_info=''
#     sanc_amt=acc_no=last_pay=curr_bal=emi=overdue_amt=pmt_feq=repay_tenure=int_rate=0
    dpd_name_arr = ['mem_name','open_date','sanc_amt','acc_no','loan_type','last_pay','curr_bal','closed_date','overdue_amt','reported_date','emi','ownership','pmt_hist_start','pmt_feq','pmt_hist_end','repay_tenure','int_rate','dpd_info']
    dpd_name_dict = {dpd_name_arr[i]:'' for i in range(len(dpd_name_arr))}
    dfs=[]
    all_dpd=[]
    for amt_idx in range(len(amt_idx_list)):
        
        dpd_iterator=0
        for i in range(amt_idx_list[amt_idx],ir_list[amt_idx]+5):
            
#             if txt_list[i].split(':')[0]=='CLOSED':
#                 dpd_name_dict[dpd_name_arr[dpd_iterator]]=txt_list[i].split(':')[1]
#                 dpd_iterator+=1
            if txt_list[i]=='DAYS PAST DUE/ASSET CLASSIFICATION (UP TO 36 MONTHS; LEFT TO RIGHT)':
                part1=[]
                other_part = []
#                 print(i+1,i+30)
                for j in range(i+1,i+80):
                    if (txt_list[j].find('ENQUIRIES') !=-1 or txt_list[j].find('ACCOUNT') !=-1 or txt_list[j].find('Services')!=-1):
#                         print(txt_list[i])
                        break
                    other_part.append(txt_list[j])
                part1.extend(other_part)
                dpd_name_dict['dpd_info']=dpd_decorator(other_part)
                dpd_iterator+=1
            else:
                for elem in dpd_elements:
                    
                    if txt_list[i].find(elem) !=-1:
#                         print(e lem,txt_list[i],txt_list[i+1])
                        elem_idx = dpd_elements.index(elem)
                        if elem_idx!=len(dpd_elements)-1:
                            dpd_name_dict[dpd_name_arr[elem_idx]]=txt_list[i].split(':')[1]
                                
                        
                
        temp_json = {
                   
            'ACCOUNT':{'member_name':dpd_name_dict['mem_name'],'account_number':dpd_name_dict['acc_no'],
                                             'TYPE':dpd_name_dict['loan_type'],'ownership':dpd_name_dict['ownership'] },
                                 'DATES':{'opened':dpd_name_dict['open_date'],'last_payment':dpd_name_dict['last_pay'],
                                          'closed':dpd_name_dict['closed_date'],'reported':dpd_name_dict['reported_date'],
                                         'pmt_hist_start':dpd_name_dict['pmt_hist_start'],'pmt_hist_end':dpd_name_dict['pmt_hist_end'] },
                                'AMOUNTS':{'sanctioned':dpd_name_dict['sanc_amt'],'current_balance':dpd_name_dict['curr_bal'],'overdue':dpd_name_dict['overdue_amt'],
                                           'emi':dpd_name_dict['emi'],'pmt_freq':dpd_name_dict['pmt_feq'],'repay_tenure':dpd_name_dict['repay_tenure'],'interest_rate':dpd_name_dict['int_rate']},
                                'DPD_INFO':dpd_name_dict['dpd_info']}
    
        
        dfs.append(temp_json)
    
    return dfs
import re 
def enquiry_decorator(enq_list):
    
    member_name=enquiry_date=enquiry_purpose=enquiry_amt=''
    len_enq_list = len(enq_list)
#     print(len_enq_list)
    enq_list_final=[]
    mem_flag =  False 
    date_flag = purpose_flag = amt_flag = False  
    i=0
    while(i<len_enq_list): 
#         print(i,enq_list[i])
        if mem_flag ==False and date_flag==False and  purpose_flag==False:
            if enq_list[i].split(' ')[0].isalpha() or bool(re.match('^[a-zA-Z]', enq_list[i])):
                member_name = enq_list[i]
                i+=1
                mem_flag=True
            else:
                member_name=''
                mem_flag=True
#             print(member_name)
        if mem_flag ==True and date_flag==False and  purpose_flag==False:
            if enq_list[i].split('-')[0].isnumeric():
                enquiry_date=enq_list[i]
                i+=1
                date_flag=True
            else:
                enquiry_date=''
                date_flag=True
            
        if mem_flag ==True and date_flag==True and purpose_flag==False:
            
                
            if enq_list[i].split(' ')[0].isalpha() or bool(re.match('^[a-zA-Z]', enq_list[i])):
#                 print(i,enq_list[i])
                enquiry_purpose=enq_list[i]
                i+=1
                purpose_flag=True
            else:
                enquiry_purpose=''
                purpose_flag=True
            
        if (mem_flag ==True) and (date_flag==True) and (purpose_flag==True):
            if enq_list[i].split(',')[0].isnumeric():
                enquiry_amt=enq_list[i]
                i+=1
                date_flag = purpose_flag = mem_flag = False
                enq_list_final.append({'member_name':member_name,'enquiry_date':enquiry_date,'enquiry_purpose':enquiry_purpose,
                                'enquiry_amount':enquiry_amt})
                
                
            else:
                enquiry_amt=''
                date_flag = purpose_flag = mem_flag = False 
                enq_list_final.append({'member_name':member_name,'enquiry_date':enquiry_date,'enquiry_purpose':enquiry_purpose,
                                'enquiry_amount':enquiry_amt})
#         if i==26:
#             break
#         print(i,member_name,enquiry_date,enquiry_purpose,enquiry_amt)    
                  
    return enq_list_final

def get_user_identifier(txt_list):
    start_idx = 0
    end_idx =4
    for i in range(len(txt_list[:50])):
        if txt_list[i].find('IDENTIFICATION TYPE')!=-1:
            start_idx=i
        if txt_list[i].find('TELEPHONE(S)') != -1:
            end_idx = i
    final_dict = {}
    for i in range(start_idx+4,end_idx-1,2):
        final_dict[txt_list[i]] = txt_list[i+1]
    if end_idx <10:
        return {}
    return final_dict
            
        
def all_attributes(txt_list):
    name=gender=dob=consumer_name=''
    for i in range(min(50,len(txt_list))): 
        if txt_list[i].find('NAME:')!=-1:
            name = txt_list[i].split(':')[1]
        if txt_list[i].find('GENDER:')!=-1:
            gender=txt_list[i].split(':')[1]
        if txt_list[i].find('DATE OF BIRTH:')!=-1:
            dob=txt_list[i].split(':')[1]
        if txt_list[i].find('CONSUMER:')!=-1:
            consumer_name = txt_list[i].split(':')[1]
    if name=='':
        name=consumer_name
#     dob_idx = txt_list.indbex('DATE  OF BIRTH:')
#     dob = txt_list[dob_idx+1]
    cibil_name_idx = txt_list.index('SCORE NAME')
    cibil_name = txt_list[cibil_name_idx+3]
    cibil_info_idx = txt_list.index('SCORING FACTORS')
    cibil_info_end_idx=30
    for i in range(min(50,len(txt_list))):
        if txt_list[i].find('POSSIBLE RANGE') != -1:
            cibil_info_end_idx=i
    cibil_info_with_factors = txt_list[cibil_info_idx+1:cibil_info_end_idx]
    # account summary 
    
    #'TOTAL:', 'OVERDUE:', 'ZERO-BALANCE:'
    summary_idx = txt_list.index('SUMMARY:')
    total =0
    overdue = 0
    zero_balance = 0
    overdue_list=[]
    for i in range(summary_idx,summary_idx+min(25,len(txt_list))):
        if txt_list[i].find('TOTAL:') != -1:
            total = txt_list[i].split(':')[1]
        if txt_list[i].find('OVERDUE:') != -1:
            overdue_list.append(txt_list[i].split(':')[1])
#             overdue = txt_list[i].split(':')[1]
        if txt_list[i].find('ZERO-BALANCE:') != -1:
            zero_balance = txt_list[i].split(':')[1]
    if len(overdue_list)>0:   
        overdue=overdue_list[0]
    account_info = get_processed_df(txt_list)
    
    ## call user identifier utility 
    user_identifier = get_user_identifier(txt_list)
    ## phone number 
    mob_idx = txt_list.index('TELEPHONE EXTENSION')
    email_idx = txt_list.index('EMAIL CONTACT(S):')
    mobile_nos=[]
    for i in range(mob_idx+2,email_idx,2):
        mobile_nos.append(txt_list[i])
    
    email_end = txt_list.index('Services provided in association with')
    email_list_1 = txt_list[email_idx+2:email_end]
    email_list = []
    for item in email_list_1:
        if item.find('@')!=-1:
            email_list.append(item)
    ## Address 
    add_idx_list = []
    for i in range(len(txt_list)):
        if txt_list[i].find('ADDRESS:') != -1:
            add_idx_list.append(i)
    address = category = res_add = date_reported = []
    cat_idx_list = []
    for i in range(min(add_idx_list),max(add_idx_list)+5):
        if txt_list[i].find('CATEGORY:') != -1:
            cat_idx_list.append(i)
    res_idx_list = []
    for i in range(min(add_idx_list),max(add_idx_list)+5):
        if txt_list[i].find('RESIDENCE CODE:') != -1:
            res_idx_list.append(i)
    date_idx_list = []
    for i in range(min(add_idx_list),max(add_idx_list)+5):
        if txt_list[i].find('DATE REPORTED:') != -1:
            date_idx_list.append(i)
    add_info=[]
    for i in range(len(add_idx_list)):
        add_info.append( {
            'address' : txt_list[add_idx_list[i]].split(':')[1],
            'category' : txt_list[cat_idx_list[i]].split(':')[1],
            'residential_code' : txt_list[res_idx_list[i]].split(':')[1],
            'date_reported' : txt_list[date_idx_list[i]].split(':')[1]
        }
        )
    ## saving all acc idx for case of 0 loans     
    all_acc_idx_list = []
    for i in range(len(txt_list)):
        if txt_list[i].find('ACCOUNT(S):')!=-1:
            all_acc_idx_list.append(i)
    final_acc_idx = max(all_acc_idx_list)
    
    enq_idx_list = []
    last_dpd_index=0
    for i in range(min(50,len(txt_list)),len(txt_list)):
        if txt_list[i]=='ENQUIRIES:' or txt_list[i].find('CONTROL NUMBER:') !=-1:
            enq_idx_list.append(i)
        if txt_list[i]=='DAYS PAST DUE/ASSET CLASSIFICATION (UP TO 36 MONTHS; LEFT TO RIGHT)':
            last_dpd_index = i
#     enq_start_idx = enq_idx_list[-1]
    # collect all possible index post dpd for enquiry
    end_of_report = len(txt_list)-1
    for i in range(last_dpd_index,end_of_report):
        if txt_list[i].find('END OF REPORT') !=-1:
            end_of_report=i
    
    
    
    final_enq_list_1=[]
    for i in range(len(enq_idx_list)):
        if enq_idx_list[i]>last_dpd_index and enq_idx_list[i]>final_acc_idx and enq_idx_list[i]<end_of_report:
            final_enq_list_1.append(enq_idx_list[i])
    final_enq_list=[]    
    ## check if min diff b/w index >=4
    flag_list_enq=[]
    for i in range(1,len(final_enq_list_1)):
        if (final_enq_list_1[i]-final_enq_list_1[i-1])<3:
            flag_list_enq.append(i-1)
#     print(flag_list_enq)
    for i in range(len(final_enq_list_1)):
        if i not in flag_list_enq:
            final_enq_list.append(final_enq_list_1[i])
            
    
    enq_end_idx = len(txt_list)-1
    enq_list=[]
#     print(final_enq_list)
    for i in range(final_enq_list[-1],len(txt_list)):
            
            if txt_list[i].find('Services') != -1 or txt_list[i].find('END OF REPORT') != -1:
                enq_end_idx=i
                break
    flag=False
#     print(enq_end_idx-final_enq_list[0]-5)
#     print(final_enq_list[0]+5,enq_end_idx)
    
    for i in range(len(final_enq_list)):
        
        st_idx = final_enq_list[i]
        for j in range(st_idx,len(txt_list)):

            if txt_list[j].find('Services') != -1 or txt_list[j].find('END OF REPORT') != -1:
                enq_end_idx=j
                break
        if (enq_end_idx-final_enq_list[i]-5)%4==0: 
            member_list=enquiry_date_list=enquiry_purpose_list=enquiry_amt_list=[]
            for idx in range(st_idx+5,enq_end_idx,4):
                enq_list.append({'member':txt_list[idx],'enquiry_date':txt_list[idx+1],'enquiry_purpose':txt_list[idx+2],'enquiry_amount':txt_list[idx+3] })
        else:
    #         print(final_enq_list[0]+5,enq_end_idx)
            enq_list = enquiry_decorator(txt_list[final_enq_list[i]+5:enq_end_idx])

    final_df= pd.DataFrame({
        'name' : [name],
        'gender' : [gender],
        'cibil_info_with_factors':[cibil_info_with_factors],
        'cibil_name':[cibil_name],
        'user_identifier':[user_identifier],
        'dob':[dob],
        'address':[add_info],
        'phone_no' : [mobile_nos],
        'email' : [email_list],
        'acc_summary' : [{'total':total,'overdue':overdue,'zero_balance':zero_balance}],
        'account_info' : [account_info],
        'enquiry' : [enq_list]
    })
    return final_df
import argparse
out_loc = '/home/ubuntu/fin_services/cibil_data/parsed_data/'
def extract_text_and_save_file(name):
    try:
        print(name)
        text_list = aws_textract_processor(name)
        fname = name.split('/')[::-1][0]
        df = all_attributes(text_list)
        print(df.shape)
        fname = fname.replace('PDF','csv')
        return df
#         out_l = out_loc+fname
#         df.to_csv(out_l,index=False)
        print(' File Done : ',fname)
    except:
         print(' Error in File :  ',name)
          
    return 'Error !'
from multiprocessing import Pool
    
##  pdf parsing via pdf-miner

from pathlib import Path
from typing import Iterable, Any
import pandas as pd

from pdfminer.high_level import extract_pages


def show_ltitem_hierarchy(o: Any, depth=0, df = pd.DataFrame()):
    """Show location and text of LTItem and all its descendants"""
    if depth == 0:
         df = pd.DataFrame()
    # print('x:' + get_indented_name(o, depth))
    if get_indented_name(o, depth) == "    LTTextBoxHorizontal":
        box_c = get_optional_bbox(o)
        x1 = int(box_c.strip().split()[0])
        y1 = int(box_c.strip().split()[1])
        x2 = int(box_c.strip().split()[2])
        y2 = int(box_c.strip().split()[3])
        df = pd.concat([df, pd.DataFrame([[get_indented_name(o, depth), get_optional_bbox(o), x1, y1, x2, y2, get_optional_text(o)]], columns = ['indented_name', 'box_coordinates', 'x1', 'y1', 'x2', 'y2', 'text'])])
    if isinstance(o, Iterable):
        for i in o:
            df = show_ltitem_hierarchy(o = i, df = df, depth=depth + 1)
                        
    return df


def get_indented_name(o: Any, depth: int) -> str:
    """Indented name of LTItem"""
    return '  ' * depth + o.__class__.__name__


def get_optional_bbox(o: Any) -> str:
    """Bounding box of LTItem if available, otherwise empty string"""
    if hasattr(o, 'bbox'):
        return ''.join(f'{i:<4.0f}' for i in o.bbox)
    return ''

def get_optional_text(o: Any) -> str:
    """Text of LTItem if available, otherwise empty string"""
    if hasattr(o, 'get_text'):
        return o.get_text().strip()
    return ''

def get_dpd_values_pdf_miner(df):
    '''
    map key value corresponding to similar key value | corresponding to same box 
    for same box delta x =1 and delta y=1
    '''
    df = df.reset_index(drop = False)
    df['dpd_flag'] = df.text.str.lower().str.contains('days past due', regex=False)
    a = df[df.dpd_flag == True]
    start_idx_list = list(a.index)
#     print(start_idx_list) 
    items_pushed = []
    all_vals = []
    for start_idx in start_idx_list:
        end_idx = df.shape[0]-1
        for i in range(start_idx,df.shape[0]):
            txt_val = df['text'][i]
            
            if (txt_val.find('ENQUIRIES') !=-1 or txt_val.find('ACCOUNT') !=-1 or txt_val.find('Services')!=-1) or txt_val.find('COPYRIGHT')!=-1:
                    end_idx=i    
                    break
#         print(start_idx,end_idx)
        dpd_dict = {}
        temp_df = df[start_idx+1:end_idx]
        for i,r in temp_df.iterrows():
            

            x_val1, x_val2 = r['x1'],r['x2']
            y_val1,y_val2 = r['y1'],r['y2']
            txt_val = r['text']
            
            k=''
            if len(txt_val.split('-'))>1:
#                 print(txt_val)
                k=txt_val
#                 print(df[(abs(df['x1']-x_val1)<2) & ((df['y1']-y_val1)<20) ]['text'].tolist())
                temp_list = temp_df[(abs(temp_df['x1']-x_val1)<2) & (temp_df['y1']< y_val1 ) ]['text'].tolist()
                curr_list =temp_list
#                 for item in temp_list:
#                     if item.split('-')[0].isnumeric() or item=='XXX':
#                         curr_list.append(item)
#                 print(curr_list)
#                 if len(curr_list)>1:
                    
                v = curr_list[0]
                items_pushed.append(k)
                dpd_dict[k]=v
    #             items_pushed.append(v)
#                     all_vals.append({k:v})
            else:
                continue
        all_vals.append(dpd_dict)
        
    return all_vals
    
    
if __name__ == "__main__":
    # Document
    
  
    parser = argparse.ArgumentParser(description ='Enter docname')
    
    # Adding Arguments
    parser.add_argument('doc',type = str)
    args = parser.parse_args()
    
    
    df_path = extract_text_and_save_file(args.doc)
    path = args.doc
    path = Path(path).expanduser()
    pages = extract_pages(path)
    df = show_ltitem_hierarchy(pages)
    list_dpd_json = get_dpd_values_pdf_miner(df)
    df_path['account_info_new'] = [list_dpd_json]
    fname = (args.doc).split('/')[::-1][0]
    fname = fname.replace('PDF','csv')
    out_l = out_loc+fname
    df_path.to_csv(out_l,index=False)
    
#     for name in glob.glob('/home/ec2-user/SageMaker/text_extraction_pdf/cibil_data/cibil_pdf_new/*'):
#         print(name)
#         extract_text_and_save_file(name)
