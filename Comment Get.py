import requests

# Specify the bug ID for which you want to retrieve comments
bug_id = "565790"  # Replace with the desired bug ID

# Define the Bugzilla REST API URL for comments
base_url = f"https://bugzilla.mozilla.org/rest/bug/{bug_id}/comment"

# Send a GET request to the API
comments_response = requests.get(base_url)

# Check if the request was successful
if comments_response.status_code == 200:
    # Parse the response JSON
    comments_data = comments_response.json()

    # Extract and print the comments
    if "bugs" in comments_data:
        bug_comments = comments_data["bugs"].get(bug_id, {}).get("comments", [])
        for comment in bug_comments:
            print(f"Comment ID: {comment['id']}")
            print(f"Comment Text: {comment['text']}")
            print(f"Creator: {comment['creator']}")
            print(f"Time: {comment['time']}\n")
else:
    print(f"Failed to retrieve comments. Status code: {comments_response.status_code}")
