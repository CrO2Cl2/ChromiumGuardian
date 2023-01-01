# This is a script to count the number of rows that contain each laber, to keep the dataset balanced, and keep track of the size of the dataset
# this script is not needed for actual use of ChromiumGuardian
import pandas as pd

# Read in the data
df = pd.read_csv('spam_data.csv', encoding='utf-8')

# Count the number of rows with the 'spam' label
spam_count = df[df['label'] == 'spam'].shape[0]

# Count the number of rows with the 'not spam' label
not_spam_count = df[df['label'] == 'not spam'].shape[0]

# Print the results
print(f"Number of rows with 'spam' label: {spam_count}")
print(f"Number of rows with 'not spam' label: {not_spam_count}")
