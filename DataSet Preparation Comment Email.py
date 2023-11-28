import csv
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import time

CommentEmailID = {}
count1 = 0
with open('Another Dataset.csv', 'r', encoding='utf-8') as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)
    csv_data = list(csv_reader)
    
    for row in csv_data:
        bug_id = row[0]  # Replace with the desired bug ID
        base_url = f"https://bugzilla.mozilla.org/rest/bug/{bug_id}/comment"
        
        try:
            # Send a GET request to the API
            comments_response = requests.get(base_url)
            comments_response.raise_for_status()  # Raise HTTPError for bad responses
            time.sleep(1)  # Add a small delay to respect API rate limits
            print(count1)
            count1=count1+1
            if comments_response.status_code == 200:
                # Parse the response JSON
                comments_data = comments_response.json()
                
                # Extract and print the comments
                if "bugs" in comments_data:
                    bug_comments = comments_data["bugs"].get(bug_id, {}).get("comments", [])
                    commentEmails = ""
                    for i in range(1, len(bug_comments)):
                        if i < len(bug_comments) and i > 1:
                            commentEmails += ","
                        commentEmails += bug_comments[i]['creator']
                    
                    CommentEmailID[bug_id] = commentEmails
        except requests.exceptions.RequestException as e:
            print(f"Error fetching comments for Bug ID {bug_id}: {e}")

# Load the existing Excel file into a pandas DataFrame
file_path = 'Another Dataset.csv'
df = pd.read_csv(file_path)

# Add a new column named 'Comment Emails' and fill it with the obtained data
df['Comment Emails'] = pd.DataFrame(CommentEmailID.values())

# Save the updated DataFrame back to the CSV file
df.to_csv('Another Dataset.csv', index=False)
