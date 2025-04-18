# GitHub Issue Summarizer CLI

A Python command-line tool to:
- Fetch child issues of a GitHub initiative
- Gather recent comments
- Summarize using OpenAI (configurable prompt)
- Allow interactive review/edits
- Post summary to a master issue

## Setup
1. Install dependencies:
   ```zsh
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your GitHub and OpenAI API keys:
   ```env
   GITHUB_TOKEN=your_github_token
   OPENAI_API_KEY=your_openai_api_key
   ```
3. Edit `instructions.txt` to customize the summary prompt.

## Usage
```zsh
python main.py --initiative https://github.com/user/repo/issues/X --comments 3
```

Where:
- `--initiative` is the URL of the initiative (parent) issue that has child issues linked to it
- `--comments` (optional) is the number of recent comments to fetch from each issue (default: 3)

## API Token Requirements

For this tool to work properly, you need:

1. **GitHub Personal Access Token**: Must have `repo` scope permissions to access issues
   - Create one at: https://github.com/settings/tokens

2. **OpenAI API Key**: Required for generating summaries
   - Create one at: https://platform.openai.com/api-keys

Both tokens should be specified in the `.env` file in the project root.
