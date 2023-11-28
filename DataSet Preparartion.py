import csv
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import requests


DescriptionText = {}
CommentEmailID = {}
count1 = 0
with open('Practice DataSet.csv', 'r', encoding='utf-8') as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)
    csv_data = list(csv_reader)
    for row in csv_data:
        bug_id = row[0]  # Replace with the desired bug ID
        # Define the Bugzilla REST API URL for comments
        base_url = f"https://bugzilla.mozilla.org/rest/bug/{bug_id}/comment"

        # Send a GET request to the API
        comments_response = requests.get(base_url)
        
        if comments_response.status_code == 200:
            # Parse the response JSON
            comments_data = comments_response.json()

            # Extract and print the comments
            if "bugs" in comments_data:
                bug_comments = comments_data["bugs"].get(bug_id, {}).get("comments", [])
                DescriptionText[bug_id] = bug_comments[0]['text']
                commentEmails = ""
                for i in range(1, len(bug_comments)):
                    if i<len(bug_comments) and i>1:
                        commentEmails += ","
                    commentEmails += bug_comments[i]['creator']
                
                CommentEmailID[bug_id] = commentEmails

print(DescriptionText)
print(CommentEmailID)
# Load the existing Excel file into a pandas DataFrame
file_path = 'Practice DataSet.csv'
df = pd.read_csv(file_path)

# Add a new column named 'NewColumn' and fill it with some data
#df['Description'] = pd.DataFrame(DescriptionText.values())
df['Comment Emails'] = pd.DataFrame(CommentEmailID.values())

# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)