"""
Global variables
"""
import praw

def init():
    global reddit
    reddit = praw.Reddit(
            client_id="ChMem9TZYJif1A",
            client_secret="3HkLZRVIBwAWbUdYExTGFK0e35d1Uw",
            user_agent="android:com.example.myredditapp:v1.2.3"
        )