# JSON File Processing Guide for SMS Phishing Detection

This guide explains how to use a directory of JSON files to run the SMS phishing detection script with the "Prompt_for_detection" functionality.

## Overview

The SMS phishing detection system can process JSON files containing SMS messages and analyze them for phishing attempts. The system uses the "Prompt_for_detection" (implemented as `_get_detection_prompt_template()`) to perform comprehensive analysis.

## Prerequisites

1. **API Keys**: Ensure you have the following API keys set in your environment:
   - `OPENAI_API_KEY`: For GPT-4o analysis
   - `JINA_READER_API_KEY`: For web content extraction
   - `GOOGLE_CLOUD_API_KEY`: For brand domain searches
   - `GOOGLE_CSE_ID`: For Google Custom Search Engine

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## JSON File Format

### Single Message Format
```json
{
  "SMS": "Your SMS message text here",
  "label": "phishing"  // Optional: "phishing", "legitimate", or "spam"
}
```

### Multiple Messages Format
```json
[
  {
    "SMS": "First SMS message",
    "label": "phishing"
  },
  {
    "SMS": "Second SMS message",
    "label": "legitimate"
  }
]
```

### SMS-Only Format (No Labels)
```json
{
  "SMS": "Your SMS message text here"
}
```

## Usage Instructions

### Step 1: Create Sample JSON Files (Optional)

Create sample JSON files for testing:

```bash
python create_sample_json_files.py
```

This creates a `sample_json_files/` directory with example JSON files.

### Step 2: Process JSON Directory

Process all JSON files in a directory:

```bash
python process_json_directory.py --input-dir /path/to/json/files --output-dir /path/to/results
```

#### Examples:

```bash
# Process sample files
python process_json_directory.py --input-dir sample_json_files --output-dir results

# Process with file limit (for testing)
python process_json_directory.py --input-dir sample_json_files --output-dir results --max-files 3

# Process your own JSON files
python process_json_directory.py --input-dir /home/user/sms_data --output-dir /home/user/analysis_results
```

### Step 3: Review Results

The script creates:
- Individual analysis directories for each message
- A summary file with processing statistics
- Detailed analysis results in JSON format

## Output Structure

```
output_dir/
├── processing_summary.json          # Overall processing summary
├── file_0_message_0/               # Analysis for first message
│   ├── analysis_output.json        # Detailed analysis results
│   └── screenshot_0.png           # Website screenshot (if URL present)
├── file_0_message_1/               # Analysis for second message
│   ├── analysis_output.json
│   └── screenshot_0.png
└── ...
```

## Analysis Results

Each `analysis_output.json` contains:

```json
{
  "is_URL": true,
  "URLs": {
    "0": {
      "URL": "https://example.com",
      "final_URL": "https://example.com",
      "redirect_chain": [...],
      "html_summary": "...",
      "domain_info": "...",
      "Image_content": "...",
      "brand_search": {...}
    }
  },
  "is_brand": true,
  "brands": ["BRAND_NAME"],
  "SMS": "Original SMS message",
  "detect_result": {
    "brand_impersonated": "BRAND_NAME",
    "URL": "https://example.com",
    "rationales": "Detailed analysis...",
    "brief_reason": "Brief reason...",
    "category": true,  // true = phishing/spam, false = legitimate
    "advice": "Safety advice..."
  },
  "user_friendly_output": "User-friendly explanation",
  "detection_prompt": "Full prompt used for detection"
}
```

## The "Prompt_for_detection" System

The detection system uses a comprehensive prompt that:

1. **Categorizes SMS types**:
   - Online gambling, bets, adult content → Phishing/Spam
   - Legitimate organization messages → Legitimate
   - Friend/family conversations → Legitimate
   - Promotions/advertisements → Spam
   - Fraudulent attempts → Phishing

2. **Analyzes multiple indicators**:
   - URL redirect chains
   - Brand impersonation
   - Website screenshots
   - HTML content analysis
   - Domain registration information
   - Google search results for brands

3. **Provides detailed rationales** and safety advice

## Advanced Usage

### Custom Processing

You can modify the processing script to:
- Filter messages by type
- Add custom analysis steps
- Integrate with other systems
- Batch process large datasets

### Integration with Existing Workflows

```python
from main import SMSPhishingDetector
from config import openai_api_key, jina_api_key, google_cloud_API_key, search_engine_ID

# Initialize detector
detector = SMSPhishingDetector(
    openai_api_key=openai_api_key,
    jina_api_key=jina_api_key,
    google_cloud_API_key=google_cloud_API_key,
    search_engine_id=search_engine_ID
)

# Process single message
result = detector.detect_sms_phishing("Your SMS message here")
```

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```
   Error: OPENAI_API_KEY not found in environment variables
   ```
   Solution: Set your API keys in environment variables or `.env` file

2. **JSON Parsing Errors**:
   ```
   Error parsing JSON file: Expecting ',' delimiter
   ```
   Solution: Validate your JSON files using a JSON validator

3. **File Not Found**:
   ```
   Error: Input directory does not exist
   ```
   Solution: Check the path to your JSON files directory

4. **Memory Issues**:
   - Process files in smaller batches using `--max-files`
   - Ensure sufficient disk space for screenshots and analysis results

### Performance Tips

1. **For Large Datasets**:
   - Use `--max-files` to process in batches
   - Monitor disk space (screenshots can be large)
   - Consider processing during off-peak hours

2. **For Testing**:
   - Start with small sample files
   - Use the sample JSON files for initial testing
   - Check API key limits and quotas

## Example Workflow

```bash
# 1. Create sample files
python create_sample_json_files.py

# 2. Process the samples
python process_json_directory.py --input-dir sample_json_files --output-dir test_results

# 3. Check results
ls test_results/
cat test_results/processing_summary.json

# 4. View individual analysis
cat test_results/file_0_message_0/analysis_output.json
```

## API Usage and Costs

The system makes API calls to:
- **OpenAI GPT-4o**: For text analysis and detection
- **Jina Reader**: For web content extraction
- **Google Custom Search**: For brand domain verification

Monitor your API usage and costs, especially for large datasets.

## Security Considerations

- The system accesses URLs in SMS messages
- Screenshots are taken of websites
- Ensure you have permission to analyze the SMS messages
- Be cautious with sensitive or personal data

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your JSON file format
3. Ensure all API keys are properly configured
4. Check the processing logs for specific error messages



