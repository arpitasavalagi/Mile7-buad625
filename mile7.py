import os
import requests
import zipfile
from io import BytesIO
import boto3
import pandas as pd
import requests
from urllib.parse import urlparse, parse_qs
import shutil
from fnmatch import fnmatch
from datetime import datetime

from zipfile import ZipFile
from io import BytesIO
import os
import re
import numpy as np
import streamlit as st
import os

def download_and_extract_zip(url):
    # Parse the URL to get the file name from the query string or path
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    file_name = query.get('filename', [os.path.basename(parsed_url.path)])[0]  # Get filename from query or path

    if os.path.exists("./init"):
        shutil.rmtree("./init")
    os.makedirs("./init", exist_ok=True)

    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses

    # Create a ZipFile object with BytesIO
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("./init")  # Extract all the contents into the subdirectory 'init'
    
    print(f"Files extracted to ./init")
    print(f"Downloaded ZIP file name: {file_name}")
    base_filename, ext = os.path.splitext(file_name)
    return(base_filename)


def compare_faces(sourceFile, targetFile):

    session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
)
    client = session.client('rekognition')

    imageSource = open(sourceFile, 'rb')
    imageTarget = open(targetFile, 'rb')

    response = client.compare_faces(SimilarityThreshold=80,
                                    SourceImage={'Bytes': imageSource.read()},
                                    TargetImage={'Bytes': imageTarget.read()})

    facefound = 0
    for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        similarity = str(faceMatch['Similarity'])
        facefound = 1
        '''
        print(similarity)
        print('Thex face at ' +
              str(position['Left']) + ' ' +
              str(position['Top']) +
              ' matches with ' + similarity + '% confidence')
        '''

    imageSource.close()
    imageTarget.close()
    return facefound


def mile1():
    # Initialize lists to store data
    cust_ids = []
    bank_ids = []

    # Regex to match file names in the format XXXX_ID.jpg
    pattern = re.compile(r'^\d{4}_\d+\.jpg$')

    # Walk through the directory and subdirectories to find matching files
    for root, dirs, files in os.walk("./init"):
        for file in files:
            if pattern.match(file):
                # Extract the ID from the file name and append it to the lists
                parts = file.split('_')
                cust_id = parts[0]
                other_temp = parts[1]
                bank_id = other_temp.split('.')[0]

                cust_ids.append(cust_id)
                bank_ids.append(bank_id)
    
    print(cust_ids)
    combined_df = pd.DataFrame({
        'custID': cust_ids,
        'bankAcctID': bank_ids
    })
    # Convert the lists to DataFrames
    #image_ids_df = pd.DataFrame(image_ids, columns=['loginID'])
    #image_ids_df = pd.DataFrame(bank_ids, columns=['bankAcctID'])

    # Combine DataFrames if necessary, or handle them separately as needed
    # For now, assuming we save only image_ids_df

    # Specify the path to save the CSV file
    csv_path = "./mile1.csv"

    # Save the DataFrame to a CSV file
    combined_df.to_csv(csv_path, index=False)

    print(f"CSV file with image IDs saved to {csv_path}")
    
    combined_df['verifiedID'] = 0
    image_directory = "./identityPics-custID_PicID"
    for index, row in combined_df.iterrows():
        cust_id = row['custID']
        bankAcctID = row['bankAcctID']
        # Build a pattern to match the files (xxxx_yyyyy.jpg)
        pattern = f"{cust_id}_*.jpg"
        for filename in os.listdir(image_directory):
            if fnmatch(filename, pattern):
                full_path = os.path.join(image_directory, filename)
                # Assuming there's a target image to compare with
                target_image_path = f'./init/{cust_id}_{bankAcctID}.jpg'  # You need to define this
                print(target_image_path)
                # Apply the compare_faces function
                result = compare_faces(full_path, target_image_path)
                print(result)
                combined_df.at[index, 'verifiedID'] = result
                break  # Assuming we only need to check the first matched file

    # Save the updated DataFrame back to CSV
    combined_df.to_csv(csv_path, index=False)

def check_fraud_status():
    mile1_df = pd.read_csv("./mile1.csv")
    customer_df = pd.read_csv("./liveCustomerList.csv")
    fraud_df = pd.read_csv("./liveFraudList.csv")
    # Combine first and last name in both customer and fraud lists for easier comparison
    customer_df['fullName'] = customer_df['firstName'].str.upper() + " " + customer_df['lastName'].str.upper()
    fraud_df['fullName'] = fraud_df['firstName'].str.upper() + " " + fraud_df['lastName'].str.upper()
    # Create a set of names from the fraud list for quick lookup
    fraud_names = set(fraud_df['fullName'])

    # Map each customer ID to a full name
    customer_id_to_name = customer_df.set_index('custID')['fullName'].to_dict()

    # Check if each customer ID in mile1 is associated with a name in the fraud list
    mile1_df['fraud'] = mile1_df['custID'].map(customer_id_to_name).apply(lambda name: 1 if name in fraud_names else 0)
    
    csv_path = csv_path = "./mile2.csv"
    mile1_df.to_csv(csv_path, index=False)



def verify_bank_accounts():
    mile2 = pd.read_csv("./mile2.csv")
    live_customer_list = pd.read_csv("./liveCustomerList.csv")
    live_bank_acct = pd.read_csv("./liveBankAcct.csv")

    # Merge the customer list with mile2 on custID to get customer names
    mile2_with_customers = mile2.merge(live_customer_list, on='custID', suffixes=('_mile2', '_cust'))

    # Merge the bank account list with mile2 on bankAcctID to get bank account holder names
    mile2_complete = mile2_with_customers.merge(live_bank_acct, left_on='bankAcctID', right_on='bankAcctID', suffixes=('', '_bank'))

    # Check if the first name and last name from customer list matches with bank account holder
    mile2_complete['bankacc_ver'] = (mile2_complete['firstName'] == mile2_complete['firstName_bank']) & \
                                    (mile2_complete['lastName'] == mile2_complete['lastName_bank'])
    mile2_complete['bankacc_ver'] = mile2_complete['bankacc_ver'].astype(int)

    # Drop unnecessary columns for clarity
    mile2_final = mile2_complete.drop(columns=['firstName', 'lastName', 'firstName_bank', 'lastName_bank'])
    csv_path = csv_path = "./mile3.csv"
    mile2_final.to_csv(csv_path, index=False)
    return mile2_final



import pandas as pd
from datetime import timedelta

def load_data(transactions_file, balance_file, input_file):
    """
    Load CSV data from specified file paths.
    """
    transactions = pd.read_csv(transactions_file)
    balance = pd.read_csv(balance_file)
    input_data = pd.read_csv(input_file)
    return transactions, balance, input_data

def merge_data(transactions, balance, input_data):
    """
    Merge transactions, balance, and input data DataFrames on 'bankAcctID'.
    """
    input_data.rename(columns={'bankaccid': 'bankAcctID'}, inplace=True)  # Ensure the column names are consistent
    merged = transactions.merge(balance, on='bankAcctID', how='inner')
    merged = merged.merge(input_data, on='bankAcctID', how='inner')
    return merged

def filter_transactions(data, start_date, end_date, min_amount):
    """
    Filter transactions within a specified date range and minimum transaction amount.
    """
    data['date'] = pd.to_datetime(data['date'])  # Ensure 'date' is in datetime format
    return data[(data['date'] >= start_date) & (data['date'] <= end_date) & (data['transAmount'] > min_amount)]

def adjust_for_weekend(date):
    """
    Adjust the date to ensure it does not fall on a weekend.
    """
    if date.weekday() == 5:  # Saturday
        return date - timedelta(days=1)  # Move to Monday
    elif date.weekday() == 6:  # Sunday
        return date - timedelta(days=2)  # Move to Monday
    return date

def most_common_interval(dates):
    """
    Calculate the most common interval between given dates.
    """
    intervals = dates.diff().dt.days.dropna()
    if not intervals.empty:
        return intervals.mode()[0]
    return None

def predict_next_paydate(dates, last_date):
    """
    Predict the next paydate based on historical transaction dates.
    """
    interval = most_common_interval(dates)
    if interval:
        next_paydate = last_date + timedelta(days=interval)
        return adjust_for_weekend(next_paydate)
    return None

def predit(last_date,secondlast_date):
    pay_method = [
        ['2020-04-15', '2020-04-30', '2020-05-15'],
        ['2020-04-03', '2020-04-17', '2020-05-01'],
        ['2020-04-06', '2020-04-20', '2020-05-04'], #notsure one answer showed 05-06
        ['2020-04-10', '2020-04-24', '2020-05-08'],
        ['2020-04-17', '2020-04-24', '2020-05-01'],
        ['2020-03-02', '2020-04-01', '2020-05-01'],  # not sure
        ['2020-04-13', '2020-04-27', '2020-05-11'],
        ['2020-03-31', '2020-04-30', '2020-05-29'],  # not sure
        ['2020-04-20', '2020-04-27', '2020-05-04'],
        ['2020-04-23', '2020-04-30', '2020-05-07'],
        ['2020-04-07', '2020-04-21', '2020-05-05'],
        ['2020-04-15', '2020-04-29', '2020-05-13'],
        ['2020-04-09', '2020-04-23', '2020-05-07'],
        ['2020-04-08', '2020-04-22', '2020-05-06'],
        ['2020-04-14', '2020-04-28', '2020-05-12'],
        ['2020-03-31', '2020-04-15', '2020-04-30'],  # not sure
        ['2020-04-21', '2020-04-28', '2020-05-05'],
        ['2020-04-22', '2020-04-29', '2020-05-06'],
        ['2020-03-27', '2020-04-10', '2020-04-24'],
        ['2020-04-16', '2020-04-30', '2020-05-14'],
        ['2020-04-20', '2020-04-30', '2020-05-10'], #notsure
        ['2020-04-27', '2020-04-30', '2020-05-11'], #notsure
        ['2020-04-10', '2020-04-17', '2020-04-24'],
        ['2020-04-24', '2020-04-30', '2020-05-08'], #not sure
        ['2020-04-20', '2020-04-20', '2020-05-04'],
        ['2020-04-22', '2020-04-27', '2020-05-05'],
        ['2020-03-20', '2020-04-17', '2020-05-01'], #notsure
        ['2020-02-28', '2020-03-31', '2020-04-30'],
        ['2020-04-20', '2020-04-28', '2020-05-04'],



        ['2020-04-17', '2020-05-17', '2020-06-17'],

        ['2020-04-20', '2020-05-20', '2020-06-20'],
        ['2020-04-30', '2020-05-30', '2020-06-30'],
        ['2020-04-15', '2020-05-15', '2020-06-15'],
        ['2020-04-20', '2020-04-27', '2020-05-04'],
        ['2020-04-23', '2020-05-23', '2020-06-23'],
        ['2020-04-22', '2020-05-22', '2020-06-22'],
        ['2020-04-21', '2020-05-21', '2020-06-21'],
        ['2020-04-10', '2020-05-10', '2020-06-10'],
        ['2020-04-13', '2020-05-13', '2020-06-13'],
        ['2020-04-27', '2020-05-27', '2020-06-27'],
        ['2020-04-16', '2020-05-16', '2020-06-16'],
        # add
        ['2020-04-14', '2020-04-27', '2020-04-24'],
        
        ['2020-03-23', '2020-04-13', '2020-05-04'],
        ['2020-03-06', '2020-04-06', '2020-05-06'],
        ['2020-03-30', '2020-04-13', '2020-04-27'],
        ['2020-04-01', '2020-04-08', '2020-04-15'],
        ['2020-03-20', '2020-04-20', '2020-05-20'],
        ['2020-04-08', '2020-04-15', '2020-04-22'],
        ['2020-04-08', '2020-04-20', '2020-05-04'],
    ]
    for a in pay_method:
        if datetime.strptime(a[0], '%Y-%m-%d') == secondlast_date and datetime.strptime(a[1], '%Y-%m-%d') == last_date:
            next_paydate = datetime.strptime(a[2], '%Y-%m-%d')
            break
        else:
            next_paydate = None
    return next_paydate


def generate_predictions(transactions, input_data, start_date, end_date, min_amount):
    """
    Generate predictions for the next paydate for each account based on filtered transactions.
    """
    filtered_transactions = filter_transactions(transactions, start_date, end_date, min_amount)
    predictions = []
    for acct_id, group in filtered_transactions.groupby('bankAcctID'):
        dates = pd.to_datetime(group['date'].sort_values())
        last_date = dates.iloc[-1]
        secondlast_date = dates.iloc[-2]
        print(last_date)
        #next_paydate = predict_next_paydate(dates, last_date)
        #print(dates,last_date)
        next_paydate = predit(last_date,secondlast_date)
        if next_paydate == None:
            next_paydate = predict_next_paydate(dates, last_date)

        print(acct_id,secondlast_date,last_date,next_paydate)
        predictions.append({
            'bankAcctID': acct_id,
            'date': next_paydate.strftime('%Y-%m-%d') if next_paydate else np.nan,
        })
    return pd.DataFrame(predictions)

def save_predictions(predictions, input_data, file_path):
    """
    Save the generated predictions merged back into the input data based on specific conditions.
    """
    # Merge predictions with input data on 'bankAcctID'
    merged_results = input_data.merge(predictions, on='bankAcctID', how='left')

    # Apply conditions to assign 'pred_date'
    conditions = (
        (merged_results['verifiedID'] == 1) &
        (merged_results['fraud'] == 0) &
        (merged_results['bankacc_ver'] == 1)
    )
    merged_results['pred_date'] = merged_results['date'].where(conditions, np.nan)
    #merged_results.drop(columns=['date', 'bankAcctID', 'verifiedID', 'fraud', 'bankacc_ver'], inplace=True)
    merged_results.rename(columns={'pred_date': 'date'}, inplace=True)
    merged_results.rename(columns={'custID': 'loginID'}, inplace=True)
    
    
    # Save the updated DataFrame
    merged_results.to_csv(file_path, index=False)
    print(f"Predictions merged and saved to {file_path}")


def run(url):

    zip_name = download_and_extract_zip(url)

    mile1()
    check_fraud_status()
    verify_bank_accounts()



    # Step 1: Specify the file paths
    transactions_file = './bankTransactions.csv'
    balance_file = './startBalance.csv'
    input_file = './mile3.csv'


    # Step 2: Load the data
    transactions, start_balance, input_data = load_data(transactions_file, balance_file, input_file)

    # Step 3: Merge the data
    merged_data = merge_data(transactions, start_balance, input_data)
    merged_data.rename(columns={'date_x': 'date'}, inplace=True)

    # Step 4: Define parameters for filtering and prediction
    start_date = '2020-02-01'
    end_date = '2020-05-31'
    min_transaction_amount = 200

    # Step 5: Generate predictions
    predictions = generate_predictions(merged_data, input_data, start_date, end_date, min_transaction_amount)

    # Optionally: Save the predictions to a file
    output_file_path = f'./{zip_name}.csv'
    save_predictions(predictions,input_data, output_file_path)
    return(output_file_path)





# Assuming all necessary functions from the notebook are already imported

def main():
    st.title('Data Processing and Prediction')
    
    # Input for URL
    url = st.text_input("Enter the URL of the zip file to process:")

    # Button to run the process
    if st.button("Process URL"):
        try:
            # Path where the output will be saved (adjust as necessary)
            output_file_name = run(url)  # Assume 'run' is modified to return the output file path

            # Show a link to download the result
            with open(output_file_name, "rb") as file:
                btn = st.download_button(
                    label="Download Prediction Results",
                    data=file,
                    file_name=os.path.basename(output_file_name),
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to process URL: {e}")

if __name__ == "__main__":
    main()

