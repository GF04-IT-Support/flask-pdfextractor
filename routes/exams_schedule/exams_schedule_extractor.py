from importlib.resources.readers import remove_duplicates
import pdfplumber
import pandas as pd
import sys
import json
import base64
import re
from io import BytesIO
import string
from datetime import datetime
    

def extract_tables_from_pdf(base64_pdf_data):
    pdf_file_path = BytesIO(base64.b64decode(base64_pdf_data))
    pdf = pdfplumber.open(pdf_file_path)
    tables = []
    exam_name = None
    exam_name_updated = False
    degree = None
    for page in pdf.pages:
        table = page.extract_table()
        df = pd.DataFrame(table[1:], columns=table[0])
        if len(df.columns) < 7:
            continue 
        all_text = page.extract_text()
        lines = all_text.split('\n')
        if 'REGULAR' in lines[3].upper() or 'PARALLEL' in lines[3].upper() or 'ACADEMIC YEAR' in lines[3].upper():
                   header_lines = lines[:4]
        else:
            header_lines = lines[:3]
        header_text = '\n'.join(header_lines)
        if any(word.upper() in header_lines[1].upper() for word in ['MBA', 'MASTER', 'MSC', 'MPHIL', 'PHD']):
            year = re.sub(r'\(.*\)|YEAR', '', header_lines[1]).strip()
            year = re.sub(r'\s/', '/', year)  
            if 'MASTER OF BUSINESS ADMINISTRATION' in year.upper():
                degree = 'MBA'
                year = year.replace('MASTER OF BUSINESS ADMINISTRATION', 'MBA')
            elif 'MASTER OF SCIENCE' in year.upper():
                degree = 'MSC'
                year = year.replace('MASTER OF SCIENCE', 'MSC')
            elif 'MASTER OF PHILOSOPHY' in year.upper():
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
        for column in df.columns:
            if df[column].isnull().all():
                df.drop(columns=[column], inplace=True)
        df['Year'] = year
        table = df.values.tolist()
        if table:
            tables.extend(table)
        if not exam_name_updated:
            if 'BACHELOR' in header_lines[1].upper():
                exam_name += ' (UNDERGRADUATE)'
                exam_name_updated = True
            elif degree:
                exam_name += ' (POSTGRADUATE)'
                exam_name_updated = True
    return tables, exam_name

def create_dataframe(tables):
    df = pd.DataFrame(tables, columns=[f"Column_{i+1}" for i in range(len(tables[0]))])
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
    
def clean_venue(venue):
    cleaned_parts = []
    parts = venue.split(', ')
    if len(parts) == 1:  
        return venue  
    for part in parts:
        words = part.split()
        for i, word in enumerate(words):
            if (len(word) == 1 and word.isupper()) or word in ['SMA', 'ACCRA', 'ONLINE','COMPUTER LABS', 'MAIN LIBRARY ICT LAB']:
                if i < len(words) - 1:
                    cleaned_parts.append(' '.join(words[:i+1]))
                else:
                    cleaned_parts.append(part)
                break
    return ', '.join(cleaned_parts)


def clean_dataframe(df):
    for col in df.columns:
        if col not in ['Venue', 'Course Code', 'Time']:
            df[col] = df[col].replace('\n', ' ', regex=True)
    df['Time'] = df['Time'].str.replace(r'\.$', '', regex=True).str.replace('.', ':')
    df['Venue'] = df['Venue'].apply(lambda x: ', '.join([re.sub(r'.*?(?=PG|SMA|SAARAH MENSAH AUD|SAARAH MENSAH AUDITORIUM)', '', line) for line in x.split('\n')]) if '\n' in x else x)
    df['Venue'] = df['Venue'].apply(lambda venue: ', '.join([re.sub(r'\b(?:GALLERY|BASEMENT|BASE|UPPER)\b', '', part).strip() for part in venue.split(', ')]))
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
    df['Venue'] = df['Venue'].apply(clean_venue)
    df['Venue'] = df['Venue'].apply(lambda x: ', '.join(map(str.strip, x.split(','))))
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

def merge_similar_rows(df):
    grouped = df.groupby(['Date', 'Start Time', 'End Time'])
    merged_data = [] 
    
    for _, group in grouped:
        if len(group) > 1:
            group.sort_index(inplace=True)
            new_row = group.iloc[0].copy()
                    
            years = set()
            exam_codes = set()
            venues = set()
                    
            for _, row in group.iterrows():
                years.add(row['Year'])
                exam_codes.update(v.strip() for v in row['Course Code'].split(','))
                venues.update(v.strip() for v in row['Venue'].split(','))
                                    
            new_row['Year'] = ' or '.join(sorted(years))
            final_exam_codes = remove_duplicates(exam_codes)
            final_exam_codes = [code.strip() for code in exam_codes]  
            new_row['Course Code'] = ', '.join(sorted(final_exam_codes))
            final_venues = remove_duplicates(venues)
            final_venues =[venue.strip() for venue in venues]
            new_row['Venue'] = ', '.join(sorted(final_venues))
           
            merged_data.append(new_row)
        else:
            merged_data.append(group.iloc[0])
    merged_df = pd.DataFrame(merged_data)
    merged_df.sort_index(inplace=True)
    return merged_df



def exams_main(base64_pdf_data):
    tables, exam_name = extract_tables_from_pdf(base64_pdf_data=base64_pdf_data)
    df = create_dataframe(tables)
    df = clean_dataframe(df)
    df = split_time_column(df)
    df = correct_date_column(df)
    df = df.drop_duplicates(subset=['Date', 'Course Code', 'Venue', 'Start Time', 'End Time'])
    df = merge_similar_rows(df)
    df.to_csv("exams.csv", index=False)
    exams_schedule = df.to_dict(orient='records')
    return {"exams_schedule": exams_schedule, "exam_name": exam_name}

if __name__ == "__main__":
    base64_pdf_data = sys.stdin.read()
    result = exams_main(base64_pdf_data)
    print(json.dumps(result))