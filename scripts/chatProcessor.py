import os
import sys

# Add the root directory of your project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
from collections import Counter
from typing import Dict, List, Tuple

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from classes.cls_poll import Poll
from interface.cls_livestream_message import LivestreamMessage

# Set up YouTube API client
scopes: List[str] = ["https://www.googleapis.com/auth/youtube.readonly"]
api_service_name: str = "youtube"
api_version: str = "v3"

# Get credentials and create an API client
flow = InstalledAppFlow.from_client_secrets_file("./cache/client_secret.json", scopes)
credentials = flow.run_local_server(port=0)  # This will open a web server for authentication
youtube = build(api_service_name, api_version, credentials=credentials)


def get_current_live_broadcast(youtube) -> Tuple[str, str]:
    active_broadcast = None
    while not active_broadcast:
        try:
            request = youtube.liveBroadcasts().list(part="snippet,status", mine=True)
            response = request.execute()

            # Filter to find the active broadcast
            active_broadcast = next((item for item in response.get("items", []) if item["status"]["lifeCycleStatus"] == "live"), None)

            if not active_broadcast:
                for i in range(60):
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"No active live broadcasts found. Retrying in {60 - i} seconds.")
                    time.sleep(1)
        except:
            pass
                
    print("Broadcasts found!")
    broadcast_id: str = active_broadcast["id"]
    live_chat_id: str = active_broadcast["snippet"]["liveChatId"]
    print(f"Broadcast ID: {broadcast_id}\nLive Chat ID: {live_chat_id}")
    return broadcast_id, live_chat_id



def save_and_return_new_messages(messages: List[LivestreamMessage], fullChat_json_path: str = "./cache/fullChat.json") -> List[LivestreamMessage]:
    os.makedirs(os.path.dirname(fullChat_json_path), exist_ok=True)  # Ensure directory exists

    existing_data = []
    if os.path.exists(fullChat_json_path):
        try:
            with open(fullChat_json_path, "r") as file:
                existing_data = json.load(file)
        except json.JSONDecodeError:
            print("Warning: Corrupt JSON file. Overwriting with new data.")

    existing_message_ids = {item["id"] for item in existing_data}

    new_data = [message for message in messages if message.id not in existing_message_ids]

    if new_data:
        new_data_dict = [message.to_dict() for message in new_data]
        # Combine existing and new data
        combined_data = existing_data + new_data_dict
        try:
            with open(fullChat_json_path, "w") as file:
                json.dump(combined_data, file, indent=4)
        except Exception as e:
            print(f"Error saving messages: {e}")
            return []  # Return an empty list to indicate failure

    return new_data


def get_chat_messages(youtube, live_chat_id: str) -> List[LivestreamMessage]:
    request = youtube.liveChatMessages().list(liveChatId=live_chat_id, part="snippet, authorDetails")
    response = request.execute()

    chat_messages: List[LivestreamMessage] = []
    for item in response["items"]:
        author: str = item["authorDetails"]["displayName"]
        message: str = item["snippet"]["displayMessage"]
        message_id: str = item["id"]  # Unique identifier for each message
        # print(f"{author}: {message}")
        chat_messages.append(LivestreamMessage(author, message, message_id))

    return chat_messages


def add_poll_votes(messages: List[LivestreamMessage]) -> None:
    poll = Poll.from_file()  # Load existing votes or initialize
    for msg in messages:
        poll.update_votes(msg.message)  # Update votes based on messages
    poll.to_file()  # Save updated votes back to file


# Main execution loop
broadcast_id, live_chat_id = get_current_live_broadcast(youtube)
while True:
    try:
        chat_messages = get_chat_messages(youtube, live_chat_id)
        new_chat_messages: List[LivestreamMessage] = save_and_return_new_messages(chat_messages)
        add_poll_votes(new_chat_messages)
        print(new_chat_messages)
        print("### Updated poll\n")
        time.sleep(60)  # Wait for 10 seconds before the next iteration
        
    except Exception as e:
        print("An error occurred: " + str(e))
        time.sleep(120)  # Wait for 10 seconds before the next iteration
        broadcast_id, live_chat_id = get_current_live_broadcast(youtube)