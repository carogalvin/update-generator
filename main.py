import os
import sys
import requests
from openai import OpenAI
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from PyInquirer import prompt
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GITHUB_API_URL = 'https://api.github.com'
INSTRUCTIONS_FILE = 'instructions.txt'
MODEL_NAME = "gpt-3.5-turbo"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- GitHub API Functions ---

def get_github_headers() -> Dict[str, str]:
    """
    Returns consistent headers for GitHub API requests.
    
    Returns:
        Dict[str, str]: Headers dictionary with authentication and API version
    """
    return {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }


def get_instructions() -> str:
    """
    Load instructions for the summarization from a file.
    
    Returns:
        str: The instructions content
        
    Raises:
        FileNotFoundError: If instructions file doesn't exist
    """
    try:
        with open(INSTRUCTIONS_FILE, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Instructions file '{INSTRUCTIONS_FILE}' not found.")
        print(f"Error: Instructions file '{INSTRUCTIONS_FILE}' not found.")
        sys.exit(1)


def github_issue_url_to_api(issue_url: str) -> str:
    """
    Converts a GitHub issue URL to the corresponding API URL.
    Example:
      https://github.com/user/repo/issues/8
      -> https://api.github.com/repos/user/repo/issues/8
    
    Args:
        issue_url (str): The URL of the GitHub issue

    Returns:
        str: The corresponding GitHub API URL

    Raises:
        ValueError: If the provided URL is not a valid GitHub issue URL
    """
    if not issue_url or not isinstance(issue_url, str):
        raise ValueError("Issue URL must be a non-empty string")
        
    parsed = urlparse(issue_url)
    if parsed.netloc != 'github.com':
        raise ValueError(f"Expected github.com URL, got: {parsed.netloc}")
        
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) >= 4 and path_parts[2] == 'issues':
        user = path_parts[0]
        repo = path_parts[1]
        issue_number = path_parts[3]
        return f"{GITHUB_API_URL}/repos/{user}/{repo}/issues/{issue_number}"
    else:
        raise ValueError("Invalid GitHub issue URL format. Expected: https://github.com/user/repo/issues/number")


def fetch_child_issues(initiative_issue_url: str) -> List[str]:
    """
    Fetches issues linked as children to the given initiative issue.
    Assumes GitHub issue linking is used ("linked issues").
    
    Args:
        initiative_issue_url (str): The URL of the initiative issue

    Returns:
        List[str]: A list of URLs of child issues
        
    Raises:
        requests.exceptions.RequestException: If API request fails
        ValueError: If response contains unexpected data format
    """
    try:
        headers = get_github_headers()
        # Get subissues
        events_url = f"{initiative_issue_url}/sub_issues"
        logger.info(f"Fetching child issues from: {events_url}")
        
        resp = requests.get(events_url, headers=headers)
        resp.raise_for_status()
        events = resp.json()
        
        if not isinstance(events, list):
            raise ValueError(f"Expected list response from GitHub API, got: {type(events)}")
            
        child_issues = [e['url'] for e in events if 'url' in e]
        logger.info(f"Found {len(child_issues)} child issues")
        return child_issues
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch child issues: {str(e)}")
        if resp.status_code == 404:
            print("Error: Could not find child issues. Verify that the initiative URL is correct.")
        elif resp.status_code == 403:
            print("Error: GitHub API rate limit exceeded or insufficient permissions.")
        else:
            print(f"Error communicating with GitHub API: {str(e)}")
        sys.exit(1)


def fetch_recent_comments(issue_url: str, n: int = 3) -> List[Dict[str, Any]]:
    """
    Fetches the most recent comments for a given issue.
    
    Args:
        issue_url (str): The URL of the issue
        n (int): The number of recent comments to fetch

    Returns:
        List[Dict[str, Any]]: A list of recent comments
        
    Raises:
        requests.exceptions.RequestException: If API request fails
    """
    try:
        headers = get_github_headers()
        comments_url = f"{issue_url}/comments"
        logger.info(f"Fetching comments from: {comments_url}")
        
        resp = requests.get(comments_url, headers=headers)
        resp.raise_for_status()
        comments = resp.json()
        
        if not isinstance(comments, list):
            raise ValueError(f"Expected list response for comments, got: {type(comments)}")
        
        # Return the n most recent comments or all if fewer than n
        return comments[-n:] if len(comments) >= n else comments
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch comments: {str(e)}")
        print(f"Error fetching comments: {str(e)}")
        return []  # Return empty list rather than crashing


def assemble_summary(issues_data: List[Dict[str, Any]], instructions: str) -> str:
    """
    Generates a summary of issues and their comments using OpenAI.
    
    Args:
        issues_data: List of dictionaries containing issue data with title and comments
        instructions: Instructions to guide the AI in generating the summary
        
    Returns:
        str: The generated summary text
        
    Raises:
        Exception: If OpenAI API call fails
    """
    try:
        # Prepare the context for the LLM
        context = ""
        for issue in issues_data:
            context += f"Issue: {issue['title']}\n"
            for c in issue.get('comments', []):
                # Safely access nested values with .get() to avoid KeyErrors
                user = c.get('user', {}).get('login', 'unknown')
                body = c.get('body', '')[:200]  # Truncate to first 200 chars
                context += f"- {user}: {body}\n"
            context += "\n"
        
        logger.info(f"Sending {len(context)} characters to OpenAI API")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": context}
            ]
        )
        
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary: {str(e)}")
        print(f"Error: Failed to generate summary: {str(e)}")
        sys.exit(1)


def post_summary(issue_url: str, summary: str) -> bool:
    """
    Posts the summary as a comment to the specified issue using the GitHub API.
    
    Args:
        issue_url: API URL for the GitHub issue
        summary: Summary text to post as a comment
    
    Returns:
        bool: True if posting was successful, False otherwise
        
    Raises:
        requests.exceptions.HTTPError: If the API request fails
    """
    try:
        headers = get_github_headers()
        comments_url = f"{issue_url}/comments"
        logger.info(f"Posting summary to {comments_url}")
        
        resp = requests.post(comments_url, headers=headers, json={"body": summary})
        resp.raise_for_status()
        
        logger.info("Summary successfully posted")
        print("Summary posted to issue.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post summary: {str(e)}")
        print(f"Error: Failed to post summary: {str(e)}")
        return False


def review_and_post_summary(initiative_url: str, summary: str, instructions: str) -> None:
    """
    Handles the interactive review process and posting of the summary.
    
    Args:
        initiative_url: API URL for the issue
        summary: The generated summary to review
        instructions: Instructions for OpenAI to use when revising
    """
    while True:
        confirm = input("Post this summary to the initiative issue? (y/n): ").strip().lower()
        if confirm == 'y':
            if post_summary(initiative_url, summary):
                logger.info("Summary posted successfully")
            break
        elif confirm == 'n':
            print("Enter instructions for how to revise the summary, or type 'done' to exit:")
            while True:
                user_instruction = input('> ')
                if user_instruction.strip().lower() == 'done':
                    print("Exiting without posting.")
                    return
                
                # Use OpenAI to revise the summary
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": f"Here is the current summary:\n{summary}\n\nPlease revise it as follows: {user_instruction}"}
                        ]
                    )
                    summary = response.choices[0].message.content
                    print("\n--- Revised Summary ---\n")
                    print(summary)
                    print("\n----------------------\n")
                    
                    confirm2 = input("Post this revised summary to the initiative issue? (y/n): ").strip().lower()
                    if confirm2 == 'y':
                        if post_summary(initiative_url, summary):
                            return
                    print("You can enter more instructions or type 'done' to exit.")
                except Exception as e:
                    logger.error(f"Failed to revise summary: {str(e)}")
                    print(f"Error: Failed to revise summary: {str(e)}")
            break
        else:
            print("Please enter 'y' or 'n'.")


def main():
    """
    Main function that orchestrates the issue summarization workflow.
    """
    # Parse command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Summarize GitHub issues for an initiative.")
    parser.add_argument('--initiative', required=True, help='URL of the initiative (parent) issue')
    parser.add_argument('--comments', type=int, default=3, help='Number of recent comments per issue to fetch')
    args = parser.parse_args()

    # Validate environment variables
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN environment variable not set.")
        print("Error: GITHUB_TOKEN is not set. Please add it to your .env file.")
        sys.exit(1)
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set.")
        print("Error: OPENAI_API_KEY is not set. Please add it to your .env file.")
        sys.exit(1)

    try:
        # Convert GitHub URL to API URL
        initiative_url = github_issue_url_to_api(args.initiative)
        
        # Load instructions for the AI
        instructions = get_instructions()
        
        # Fetch child issues
        print("Fetching child issues...")
        child_issues = fetch_child_issues(initiative_url)
        
        if not child_issues:
            print("No child issues found. Please verify the initiative issue URL.")
            sys.exit(0)
        
        # Gather data from child issues
        print(f"Fetching data from {len(child_issues)} child issues...")
        issues_data = []
        for url in child_issues:
            try:
                issue_resp = requests.get(url, headers=get_github_headers())
                issue_resp.raise_for_status()
                issue = issue_resp.json()
                comments = fetch_recent_comments(url, n=args.comments)
                issues_data.append({'title': issue['title'], 'comments': comments})
                print(f"• {issue['title']}")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {url}: {str(e)}")
                print(f"Warning: Failed to process issue {url}")
        
        if not issues_data:
            print("Could not collect data from any issues. Exiting.")
            sys.exit(1)
        
        # Generate summary
        print("Generating summary...")
        summary = assemble_summary(issues_data, instructions)
        print("\n--- Generated Summary ---\n")
        print(summary)
        print("\n------------------------\n")

        # Interactive review and posting
        review_and_post_summary(initiative_url, summary, instructions)
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
