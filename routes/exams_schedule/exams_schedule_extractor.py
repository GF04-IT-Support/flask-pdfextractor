import pdfplumber
import pandas as pd
import sys
import json
import base64
import re
from io import BytesIO
import string
    

def extract_tables_from_pdf(base64_pdf_data):
    pdf_file_path = BytesIO(base64.b64decode(base64_pdf_data))
    pdf = pdfplumber.open(pdf_file_path)
    tables = []
    exam_name = None
    exam_name_updated = False
    degree = None
    for page in pdf.pages:
        all_text = page.extract_text()
        lines = all_text.split('\n')
        if 'REGULAR' in lines[3] or 'PARALLEL' in lines[3] or 'ACADEMIC YEAR' in lines[3]:
            header_lines = lines[:4]
        else:
            header_lines = lines[:3]
        header_text = '\n'.join(header_lines)
        if any(word in header_lines[1] for word in ['MBA', 'MASTER', 'MSC', 'MPHIL', 'PHD']):
            year = re.sub(r'\(.*\)|YEAR', '', header_lines[1]).strip()
            year = re.sub(r'\s/', '/', year)  
            if 'MASTER OF BUSINESS ADMINISTRATION' in year:
                degree = 'MBA'
                year = year.replace('MASTER OF BUSINESS ADMINISTRATION', 'MBA')
            elif 'MASTER OF SCIENCE' in year:
                degree = 'MSC'
                year = year.replace('MASTER OF SCIENCE', 'MSC')
            elif 'MASTER OF PHILOSOPHY' in year:
                degree = 'MPHIL'
                year = year.replace('MASTER OF PHILOSOPHY', 'MPHIL')
            else:
                degree = header_lines[1]
        else:
            year = re.search(r'\d+', header_text.split('\n')[1]).group()
            if len(header_lines) == 4 and not ('REGULAR' in header_lines[3] or 'PARALLEL' in header_lines[3]):
                year += ' (' + header_lines[3] + ')'
        if not exam_name:
            if len(header_lines) == 4 and not ('REGULAR' in header_lines[3] or 'PARALLEL' in header_lines[3]):
                exam_name = header_lines[2] + ' ' + header_lines[3]
            else:
                exam_name = header_lines[2]
        table = page.extract_table()
        df = pd.DataFrame(table[1:], columns=table[0])
        df['Year'] = year
        table = df.values.tolist()
        if table:
            tables.extend(table)
        if not exam_name_updated:
            if 'BACHELOR' in header_lines[1]:
                exam_name += ' (UNDERGRADUATE)'
                exam_name_updated = True
            elif degree:
                exam_name += ' (POSTGRADUATE)'
                exam_name_updated = True
    return tables, exam_name


def create_dataframe(tables):
    df = pd.DataFrame(tables[1:], columns=tables[0])
    df.columns = ['Day/Date', 'Course Code', 'Course Name', 'No of Students', 'Time', 'Venue', 'Examiner', 'Year']
    df.drop(columns=['Examiner', 'Course Name'], inplace=True)
    df = df.fillna(method='ffill')
    df.columns = [col.replace('\n', ' ') for col in df.columns]
    return df

def clean_date(date_str):
    split_result = date_str.split(' ', 1)
    if len(split_result) > 1:
        return split_result[1].strip('/').strip()
    else:
        return date_str.strip('/').strip()

def clean_dataframe(df):
    for col in df.columns:
        if col not in ['Venue', 'Course Code']:
            df[col] = df[col].replace('\n', ' ', regex=True)
    df['Venue'] = df['Venue'].apply(lambda x: ', '.join([re.sub(r'.*?(?=PG|SMA|SAARAH MENSAH AUD|SAARAH MENSAH AUDITORIUM)', '', line) for line in x.split('\n')]) if '\n' in x else x)
    df['Venue'] = df['Venue'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation.replace(',', ''))))  
    df['Course Code'] = df['Course Code'].replace('\n(?=\d)', ' ', regex=True)
    df['Course Code'] = df['Course Code'].replace('\n', ', ', regex=True)
    df['Course Code'] = df['Course Code'].replace('(?<=[a-zA-Z])(?=\d)', ' ', regex=True)
    df.replace(to_replace=r'\n', value=' ', regex=True, inplace=True)
    df = df[~(df == df.columns).sum(axis=1).gt(1)]
    df['Day/Date'] = df['Day/Date'].apply(clean_date)
    df.rename(columns={'Day/Date': 'Date'}, inplace=True)
    df['Course Code'] = df['Course Code'].apply(lambda x: ', '.join(word.strip() for word in x.split(',')))
    df['Venue'] = df['Venue'].replace({'SAARAH MENSAH AUD': 'SMA', 'SAARAH MENSAH AUDITORIUM': 'SMA', 'BLK': 'BLOCK'}, regex=True)
    return df

def split_time_column(df):
    df['Time'] = df['Time'].str.replace('–', '-', regex=False)
    df['Time'] = df['Time'].str.lower()
    time_index = df.columns.get_loc('Time')
    time_df = df['Time'].str.split('-', expand=True)
    time_df.columns = ['Start Time', 'End Time']
    df = df.drop('Time', axis=1)
    df = pd.concat([df.iloc[:, :time_index], time_df, df.iloc[:, time_index:]], axis=1)
    df['Start Time'] = df['Start Time'].str.strip()
    df['End Time'] = df['End Time'].str.strip()
    df['Start Time'] = df.apply(lambda row: row['Start Time'] + row['End Time'][-2:] if 'am' not in row['Start Time'] and 'pm' not in row['Start Time'] else row['Start Time'], axis=1)
    df['Start Time'] = df['Start Time'].apply(lambda x: x if ':' in x else x.replace('am', ':00am').replace('pm', ':00pm'))
    df['End Time'] = df['End Time'].apply(lambda x: x if ':' in x else x.replace('am', ':00am').replace('pm', ':00pm'))
    df['Start Time'] = df['Start Time'].str.replace(' :', ':').str.replace(' am', 'am').str.replace(' pm', 'pm')
    df['End Time'] = df['End Time'].str.replace(' :', ':').str.replace(' am', 'am').str.replace(' pm', 'pm')
    return df

def correct_date_column(df):
    df[['Day', 'Month', 'Years']] = df['Date'].str.split('/', expand=True)
    df['Day'] = df['Day'].str.strip()  
    df['Month'] = df['Month'].str.strip()  
    df['Day'] = df['Day'].apply(lambda x: '0' + x if len(x) == 1 else x) 
    for i in range(1, len(df)):
        if len(df.loc[i, 'Month']) == 1:
            j = i - 1
            while len(df.loc[j, 'Month']) == 1 and j > 0:
                j -= 1
            df.loc[i, 'Month'] = df.loc[j, 'Month']

    df['Date'] = df['Day'] + '/' + df['Month'] + '/' + df['Years']
    df = df.drop(['Day', 'Month', 'Years'], axis=1)
    return df

def exams_main(base64_pdf_data):
    tables, exam_name = extract_tables_from_pdf(base64_pdf_data=base64_pdf_data)
    df = create_dataframe(tables)
    df = clean_dataframe(df)
    df = split_time_column(df)
    df = correct_date_column(df)
    df = df.drop_duplicates(subset=['Date', 'Course Code', 'Venue', 'Start Time', 'End Time'])
    exams_schedule = df.to_dict(orient='records')
    return {"exams_schedule": exams_schedule, "exam_name": exam_name}

if __name__ == "__main__":
    base64_pdf_data = sys.stdin.read()
    result = exams_main(base64_pdf_data)
    print(json.dumps(result))