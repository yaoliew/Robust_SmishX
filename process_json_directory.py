#!/usr/bin/env python3
"""
Script to process a directory of JSON files containing SMS messages
for phishing detection using the Robust SmishX system.

Usage:
    python process_json_directory.py --input-dir /path/to/json/files --output-dir /path/to/results
"""

import json
import os
import argparse
import glob
from typing import List, Dict, Any
from main import SMSPhishingDetector
from config import openai_api_key, jina_api_key, google_cloud_API_key, search_engine_ID


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON file and return list of SMS messages.
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        List[Dict]: List of SMS message dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Single message
            return [data]
        else:
            print(f"Warning: Unexpected JSON structure in {file_path}")
            return []
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def process_sms_message(detector: SMSPhishingDetector, sms_data: Dict[str, Any], 
                       output_dir: str, file_index: int, message_index: int) -> Dict[str, Any]:
    """
    Process a single SMS message for phishing detection.
    
    Args:
        detector: SMSPhishingDetector instance
        sms_data: Dictionary containing SMS data
        output_dir: Output directory for results
        file_index: Index of the source file
        message_index: Index of message within the file
        
    Returns:
        Dict: Analysis results
    """
    sms_text = sms_data.get('SMS', '')
    if not sms_text:
        print(f"Warning: No SMS text found in message {message_index}")
        return {"error": "No SMS text found"}
    
    # Create unique output directory for this message
    message_output_dir = os.path.join(output_dir, f"file_{file_index}_message_{message_index}")
    
    try:
        # Run phishing detection
        is_phishing = detector.detect_sms_phishing(
            sms_message=sms_text,
            output_dir=message_output_dir,
            enable_redirect_chain=True,
            enable_brand_search=True,
            enable_screenshot=True,
            enable_html_content=True,
            enable_domain_info=True
        )
        
        # Load the analysis results
        analysis_file = os.path.join(message_output_dir, "analysis_output.json")
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
        else:
            analysis_results = {"error": "Analysis file not found"}
        
        return {
            "original_data": sms_data,
            "is_phishing": is_phishing,
            "analysis_results": analysis_results,
            "output_directory": message_output_dir
        }
        
    except Exception as e:
        print(f"Error processing SMS message: {e}")
        return {
            "original_data": sms_data,
            "error": str(e),
            "output_directory": message_output_dir
        }


def process_json_directory(input_dir: str, output_dir: str, max_files: int = None) -> Dict[str, Any]:
    """
    Process all JSON files in a directory for SMS phishing detection.
    
    Args:
        input_dir (str): Directory containing JSON files
        output_dir (str): Directory to save results
        max_files (int): Maximum number of files to process (None for all)
        
    Returns:
        Dict: Summary of processing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = SMSPhishingDetector(
        openai_api_key=openai_api_key,
        jina_api_key=jina_api_key,
        google_cloud_API_key=google_cloud_API_key,
        search_engine_id=search_engine_ID
    )
    
    # Find all JSON files
    json_pattern = os.path.join(input_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return {"error": "No JSON files found"}
    
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    all_results = []
    total_messages = 0
    successful_messages = 0
    error_messages = 0
    
    for file_index, json_file in enumerate(json_files):
        print(f"\nProcessing file {file_index + 1}/{len(json_files)}: {os.path.basename(json_file)}")
        
        # Load messages from JSON file
        messages = load_json_file(json_file)
        if not messages:
            print(f"No valid messages found in {json_file}")
            continue
        
        file_results = {
            "source_file": json_file,
            "messages": []
        }
        
        # Process each message in the file
        for message_index, sms_data in enumerate(messages):
            total_messages += 1
            print(f"  Processing message {message_index + 1}/{len(messages)}")
            
            result = process_sms_message(
                detector, sms_data, output_dir, file_index, message_index
            )
            
            file_results["messages"].append(result)
            
            if "error" in result:
                error_messages += 1
            else:
                successful_messages += 1
        
        all_results.append(file_results)
    
    # Create summary
    summary = {
        "processing_summary": {
            "total_files": len(json_files),
            "total_messages": total_messages,
            "successful_messages": successful_messages,
            "error_messages": error_messages,
            "success_rate": (successful_messages / total_messages * 100) if total_messages > 0 else 0
        },
        "results": all_results
    }
    
    # Save summary
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete!")
    print(f"Total files: {len(json_files)}")
    print(f"Total messages: {total_messages}")
    print(f"Successful: {successful_messages}")
    print(f"Errors: {error_messages}")
    print(f"Success rate: {summary['processing_summary']['success_rate']:.1f}%")
    print(f"Results saved to: {output_dir}")
    
    return summary


def main():
    """Main function to handle command line arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Process a directory of JSON files for SMS phishing detection"
    )
    parser.add_argument(
        "--input-dir", 
        required=True,
        help="Directory containing JSON files with SMS messages"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--max-files", 
        type=int,
        help="Maximum number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    # Check API keys
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return 1
    
    if not jina_api_key:
        print("Error: JINA_READER_API_KEY not found in environment variables")
        return 1
    
    # Process the directory
    try:
        summary = process_json_directory(args.input_dir, args.output_dir, args.max_files)
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())



