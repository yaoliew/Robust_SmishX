#!/usr/bin/env python3
"""
Demonstration script showing how to use JSON files with the SMS phishing detection system.
"""

import os
import json
import tempfile
import shutil
from process_json_directory import process_json_directory
from create_sample_json_files import create_sample_json_files


def run_demo():
    """Run a complete demonstration of JSON file processing."""
    print("=" * 60)
    print("SMS Phishing Detection - JSON Processing Demo")
    print("=" * 60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        
        print(f"\n1. Creating sample JSON files in: {input_dir}")
        create_sample_json_files(input_dir)
        
        # List created files
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        print(f"   Created {len(json_files)} JSON files:")
        for file in json_files:
            print(f"   - {file}")
        
        print(f"\n2. Processing JSON files...")
        print(f"   Input directory: {input_dir}")
        print(f"   Output directory: {output_dir}")
        
        try:
            # Process the JSON files
            summary = process_json_directory(input_dir, output_dir, max_files=3)
            
            print(f"\n3. Processing Results:")
            print(f"   Total files processed: {summary['processing_summary']['total_files']}")
            print(f"   Total messages: {summary['processing_summary']['total_messages']}")
            print(f"   Successful analyses: {summary['processing_summary']['successful_messages']}")
            print(f"   Errors: {summary['processing_summary']['error_messages']}")
            print(f"   Success rate: {summary['processing_summary']['success_rate']:.1f}%")
            
            # Show sample results
            print(f"\n4. Sample Analysis Results:")
            if summary['results']:
                first_file = summary['results'][0]
                if first_file['messages']:
                    first_message = first_file['messages'][0]
                    if 'analysis_results' in first_message:
                        analysis = first_message['analysis_results']
                        print(f"   SMS: {analysis.get('SMS', 'N/A')[:50]}...")
                        print(f"   Is Phishing: {analysis.get('detect_result', {}).get('category', 'N/A')}")
                        print(f"   Brief Reason: {analysis.get('detect_result', {}).get('brief_reason', 'N/A')}")
                        print(f"   User-Friendly Output: {analysis.get('user_friendly_output', 'N/A')[:100]}...")
            
            print(f"\n5. Files created in output directory:")
            if os.path.exists(output_dir):
                output_files = os.listdir(output_dir)
                for file in output_files:
                    print(f"   - {file}")
            
        except Exception as e:
            print(f"Error during processing: {e}")
            return False
    
    print(f"\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    return True


def show_json_format_examples():
    """Show examples of different JSON formats."""
    print("\n" + "=" * 60)
    print("JSON File Format Examples")
    print("=" * 60)
    
    examples = {
        "Single Message": {
            "SMS": "[US POSTAL] Your package is ready for delivery. Confirm your address to avoid returns: https://dik.si/postal",
            "label": "phishing"
        },
        "Multiple Messages": [
            {
                "SMS": "Your Uber ride is arriving in 3 minutes.",
                "label": "legitimate"
            },
            {
                "SMS": "URGENT: Your bank account will be suspended. Click here: http://fake-bank.com",
                "label": "phishing"
            }
        ],
        "SMS Only (No Labels)": {
            "SMS": "Get 50% off all items! Limited time offer. Shop now!"
        }
    }
    
    for format_name, example in examples.items():
        print(f"\n{format_name}:")
        print(json.dumps(example, indent=2))


def show_usage_instructions():
    """Show usage instructions."""
    print("\n" + "=" * 60)
    print("Usage Instructions")
    print("=" * 60)
    
    instructions = [
        "1. Prepare your JSON files with SMS messages",
        "2. Set up your API keys (OPENAI_API_KEY, JINA_READER_API_KEY, etc.)",
        "3. Run the processing script:",
        "   python process_json_directory.py --input-dir /path/to/json/files --output-dir /path/to/results",
        "4. Review the analysis results in the output directory",
        "5. Check the processing_summary.json for overall statistics"
    ]
    
    for instruction in instructions:
        print(f"   {instruction}")
    
    print(f"\nExample commands:")
    print(f"   # Create sample files")
    print(f"   python create_sample_json_files.py")
    print(f"   ")
    print(f"   # Process sample files")
    print(f"   python process_json_directory.py --input-dir sample_json_files --output-dir results")
    print(f"   ")
    print(f"   # Process with file limit")
    print(f"   python process_json_directory.py --input-dir sample_json_files --output-dir results --max-files 5")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SMS Phishing Detection JSON Processing Demo")
    parser.add_argument("--demo", action="store_true", help="Run the complete demo")
    parser.add_argument("--examples", action="store_true", help="Show JSON format examples")
    parser.add_argument("--usage", action="store_true", help="Show usage instructions")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.examples:
        show_json_format_examples()
    elif args.usage:
        show_usage_instructions()
    else:
        print("SMS Phishing Detection - JSON Processing Demo")
        print("\nAvailable options:")
        print("  --demo      Run the complete demonstration")
        print("  --examples  Show JSON format examples")
        print("  --usage     Show usage instructions")
        print("\nExample: python demo_json_processing.py --demo")


if __name__ == "__main__":
    main()



