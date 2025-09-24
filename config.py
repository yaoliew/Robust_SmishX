import os

# Optionally load environment variables from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# OpenAI API Key can be obtained from https://platform.openai.com/api-keys
# Used for accessing GPT-4o/GPT models
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

# Jina Reader API Key can be obtained from https://jina.ai/reader/
# Used for accessing html plain text of web pages
jina_api_key = os.getenv("JINA_READER_API_KEY", "").strip()

# Google Cloud API Key can be obtained from https://console.cloud.google.com/apis/credentials
# Used for accessing Google Search API
google_cloud_API_key = os.getenv("GOOGLE_CLOUD_API_KEY", "").strip()

# Search Engine ID can be obtained from https://programmablesearchengine.google.com/controlpanel/all
search_engine_ID = os.getenv("GOOGLE_CSE_ID", "").strip()

# HTTP User-Agent header that mimics a Chrome browser on Windows 10.
http_request_header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
