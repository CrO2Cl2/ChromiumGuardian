#DATABASETOOL.PY
# script used to manually input data into the database
# This script is not needed for actual use

import csv
import os

# Read in the label
label = input("Enter the label (spam or not spam): ")

# Open a CSV file for writing in append mode
with open('spam_data.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Check if the file is empty
    if os.stat('spam_data.csv').st_size == 0:
        # If the file is empty, write the header row
        writer.writerow(['text', 'label'])
    
    # Loop indefinitely
    while True:
        # Read in the message
        message = input("Enter the message: ")
        
        # Write the message and label as a row
        writer.writerow([message, label])
        
