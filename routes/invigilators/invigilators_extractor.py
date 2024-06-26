import pdfplumber
import pandas as pd
import numpy as np
import re
import json
import base64
from io import BytesIO
import sys
from fuzzywuzzy import fuzz
from itertools import combinations
from datetime import datetime

def extract_tables_from_pdf(base64_pdf_data):
    pdf_file_path = BytesIO(base64.b64decode(base64_pdf_data))
    pdf = pdfplumber.open(pdf_file_path)
    tables = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.extend(table)
    return tables

def clean_venue(venue):
    venue = re.sub(r'[-.]', '', venue).strip()
    venue = re.sub(r'\s+', ' ', venue)
    venue = re.sub(r'.*?(?=PG|SMA|SAARAH MENSAH AUD|SAARAH MENSAH AUDITORIUM|BLOCK|BLK)', '', venue)
    venue = re.sub(r'\d', '', venue)
    venue = re.sub(r'\bBLK\b', 'BLOCK', venue)
    if venue.strip().startswith('BLOCK'):
        venue = 'PG ' + venue
    venue = re.sub(r'\bSAARAH MENSAH AUD\b', 'SMA', venue)
    venue = re.sub(r'\bSAARAH MENSAH AUDITORIUM\b', 'SMA', venue)
    venue = re.sub(r'[^\w\s]', '', venue)
    return venue.strip()

def clean_date(date_str):
    split_result = date_str.split(' ', 1)
    if len(split_result) > 1:
        return split_result[1].strip('/').strip()
    else:
        return date_str.strip('/').strip()

def clean_dataframe(df):
    df.drop(columns=['Course Name'], inplace=True)
    df.replace("", np.nan, inplace=True)
    df = df.fillna(method='ffill')
    df.columns = [col.replace('\n', ' ') for col in df.columns]
    for col in df.columns:
        if col not in ['Venue', 'Course Code']:
            df[col] = df[col].replace('\n', ' ', regex=True)
    df['Venue'] = df['Venue'].replace('(?!\nACCRA)\n', ' - ', regex=True)
    df['Venue'] = df['Venue'].apply(lambda venue: ', '.join([re.sub(r'\b(?:GALLARY|GALLERY|1ST|1ST GALLARY|1ST GALLERY|BASEMENT|BASE|UPPER|FF01|FF)\b', '', part).strip() for part in venue.split(', ')]))
    df['Time'] = df['Time'].str.replace(r'\.$', '', regex=True).str.replace('.', ':')
    df = df.assign(Venue=df['Venue'].str.split('\n')).explode('Venue')
    df['Venue'] = df['Venue'].apply(clean_venue)
    df['Course Code'] = df['Course Code'].replace('\n(?=\d)', ' ', regex=True)
    df['Course Code'] = df['Course Code'].replace('\n', ', ', regex=True)
    df['Course Code'] = df['Course Code'].replace('(?<=[a-zA-Z])(?=\d)', ' ', regex=True)
    df.replace(to_replace=r'\n', value=' ', regex=True, inplace=True)
    df = df[~(df == df.columns).sum(axis=1).gt(1)]
    df['Day/Date'] = df['Day/Date'].apply(clean_date)
    df.rename(columns={'Day/Date': 'Date'}, inplace=True)
    return df

def split_and_transform_time(time_str):
    time_str = time_str.lower()

    parts = time_str.split('-')
    start_time, end_time = None, None

    if len(parts) == 2:
        start_time, end_time = parts[0].strip(), parts[1].strip()
    elif len(parts) == 3:
        if ":" in parts[0]:
            start_time = parts[0].strip()
            end_time = parts[1].strip() + ":" + parts[2].strip()
        else:
            start_time = parts[0].strip() + ":" + parts[1].strip()
            end_time = parts[2].strip()
    

    start_time = start_time.replace(' ', '')
    end_time = end_time.replace(' ', '')
    start_time = start_time.replace('am', '').replace('pm', '')
    end_time = end_time.replace('am', '').replace('pm', '')

    if ":" not in start_time:
        start_time += ":00"
    if ":" not in end_time:
        end_time += ":00"

    start_hour = int(start_time.split(':')[0])
    if 8 <= start_hour < 12:
        start_time += 'am'
    else:
        start_time += 'pm'

    end_hour = int(end_time.split(':')[0])
    if 8 <= end_hour < 12:
        end_time += 'am'
    else:
        end_time += 'pm'

    if 'pm' in start_time and 'am' in end_time:
        end_time = end_time.replace('am', 'pm')

    try:
        start_time_obj = datetime.strptime(start_time, '%I:%M%p')
        end_time_obj = datetime.strptime(end_time, '%I:%M%p')
        start_time = start_time_obj.strftime('%I:%M%p').lower()
        end_time = end_time_obj.strftime('%I:%M%p').lower()
    except ValueError:
        pass

    return start_time, end_time

def split_time_column(df):
    df['Time'] = df['Time'].str.replace('–', '-', regex=False)
    df['Time'] = df['Time'].str.lower()
    time_index = df.columns.get_loc('Time')
    df['Start Time'], df['End Time'] = zip(*df['Time'].apply(split_and_transform_time))
    df['Start Time'] = df['Start Time'].str.replace(' ', '').apply(lambda x: ':'.join(str(int(i)).zfill(2) if i.isdigit() else i for i in x.split(':')))
    df['End Time'] = df['End Time'].str.replace(' ', '').apply(lambda x: ':'.join(str(int(i)).zfill(2) if i.isdigit() else i for i in x.split(':')))
    df = df.drop('Time', axis=1)
    df.insert(time_index, 'Start Time', df.pop('Start Time'))
    df.insert(time_index + 1, 'End Time', df.pop('End Time'))
    return df

def clean_invigilators(df):
    df['Invigilators'] = df['Invigilators'].str.replace(r'([a-z])([A-Z]\.)', r'\1 \2', regex=True)
    df['Invigilators'] = df['Invigilators'].str.replace(r'\b([A-Z])\b(?!\.)', r'\1.', regex=True)
    df['Invigilators'] = df['Invigilators'].str.replace(r'(\b\w+\b)(\s[A-Z]\.)', r'\1, \2', regex=True).str.title()
    df['Invigilators'] = df['Invigilators'].str.replace(r'\s*-\s*', '-', regex=True)
    df['Invigilators'] = df['Invigilators'].str.replace(r'[/;]', ', ', regex=True)
    df['Invigilators'] = df['Invigilators'].str.replace(r',\s*,', ',', regex=True)
    df = df.assign(Invigilators=df['Invigilators'].str.split(pat=',\s*', expand=False)).explode('Invigilators')
    df['Invigilators'] = df['Invigilators'].str.replace(r'[^\w\s.-]', '', regex=True)
    df['Invigilators_sorted'] = df['Invigilators'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
    
    replaced_invigilators = df['Invigilators'].tolist()
    for i, invigilator in enumerate(df['Invigilators']):
        for j, other_invigilator in enumerate(df['Invigilators_sorted']):
            if invigilator and other_invigilator and i != j and abs(len(invigilator) - len(other_invigilator)) <= 2:
                if fuzz.ratio(invigilator, other_invigilator) > 92:
                    replaced_invigilators[i] = df['Invigilators'].iloc[j]

    df['Invigilators'] = replaced_invigilators
    df = df.drop(columns='Invigilators_sorted')
    return df

def group_by_invigilator(df):
    df = df.drop(columns='No. of Students')
    df['NormalizedInvigilator'] = df['Invigilators'].apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
    df['Details'] = df.apply(lambda row: {col: row[col] for col in df.columns if col not in ['Invigilators', 'NormalizedInvigilator']}, axis=1)

    invigilators = df['NormalizedInvigilator'].unique()
    similar_invigilators = {}
    for invig1, invig2 in combinations(invigilators, 2):
        if abs(len(invig1) - len(invig2)) <= 4:
            if fuzz.token_sort_ratio(invig1, invig2) > 92:  
                similar_invigilators[invig2] = invig1

    df['NormalizedInvigilator'] = df['NormalizedInvigilator'].replace(similar_invigilators)

    invigilators = df['NormalizedInvigilator'].unique()
    for invig1, invig2 in combinations(invigilators, 2):
        if sorted(invig1) == sorted(invig2):  
            similar_invigilators[invig2] = invig1

    df['NormalizedInvigilator'] = df['NormalizedInvigilator'].replace(similar_invigilators)
    grouped_df = df.groupby('NormalizedInvigilator').agg({'Details': list, 'Invigilators': 'first'}).reset_index()
    return grouped_df[['Invigilators', 'Details']]


def correct_date_column(df):
    df[['Day', 'Month', 'Years']] = df['Date'].str.split('/', expand=True)
    df['Day'] = df['Day'].str.strip()  
    df['Month'] = df['Month'].str.strip()  
    df['Day'] = df['Day'].apply(lambda x: '0' + x if len(x) == 1 else x) 
    df['Month'] = df['Month'].apply(lambda x: x[:2] if len(x) > 2 else x)
    df.reset_index(drop=True, inplace=True)
    for i in range(1, len(df)):
        if len(df.loc[i, 'Month']) == 1:
            j = i - 1
            while len(df.loc[j, 'Month']) == 1 and j > 0:
                j -= 1
            df.loc[i, 'Month'] = df.loc[j, 'Month']

    df['Date'] = df['Day'] + '/' + df['Month'] + '/' + df['Years']
    df = df.drop(['Day', 'Month', 'Years'], axis=1)
    return df

def invigilators_main(base64_pdf_data):
    tables = extract_tables_from_pdf(base64_pdf_data=base64_pdf_data)
    column_headers = ['Day/Date', 'Course Code', 'Course Name', 'No. of Students', 'Time', 'Venue', 'Invigilators']
    df = pd.DataFrame(tables[1:], columns=column_headers)
    df = clean_dataframe(df)
    df = split_time_column(df)
    df = correct_date_column(df)
    df = clean_invigilators(df)
    df = df[df['Invigilators'].str.strip() != '']
    grouped_df = group_by_invigilator(df)
    grouped_df = grouped_df.rename(columns={"Invigilators": "Invigilator"})
    # grouped_df.to_csv("invigilators.csv", index=False)
    invigilators_schedule = grouped_df.to_dict(orient='records')
    return {"invigilators_schedule": invigilators_schedule}

if __name__ == "__main__":
    base64_pdf_data = sys.stdin.read()
    result = invigilators_main(base64_pdf_data)
    print(json.dumps(result))