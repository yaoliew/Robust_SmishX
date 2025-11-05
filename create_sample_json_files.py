#!/usr/bin/env python3
"""
Script to create sample JSON files for testing the SMS phishing detection system.
"""

import json
import os


def create_sample_json_files(output_dir: str = "sample_json_files"):
    """
    Create sample JSON files with SMS messages for testing.
    
    Args:
        output_dir (str): Directory to create sample files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample SMS messages
    sample_messages = [
        # Phishing examples
        {
            "SMS": "[US POSTAL] Your package is ready for delivery. Confirm your address to avoid returns: https://dik.si/postal",
            "label": "phishing"
        },
        {
            "SMS": "URGENT: Your bank account will be suspended. Click here to verify: http://fake-bank-verify.com",
            "label": "phishing"
        },
        {
            "SMS": "Congratulations! You've won $1000! Claim your prize at: https://suspicious-lottery.com/claim",
            "label": "phishing"
        },
        {
            "SMS": "Amazon: Your order #12345 has been delayed. Update payment info: http://amazon-payment-update.fake",
            "label": "phishing"
        },
        
        # Legitimate examples
        {
            "SMS": "Your Uber ride with John is arriving in 3 minutes. Driver: John D. - Honda Civic (ABC123)",
            "label": "legitimate"
        },
        {
            "SMS": "Your verification code is 123456. Do not share this code with anyone.",
            "label": "legitimate"
        },
        {
            "SMS": "FedEx: Package delivered to front door at 2:30 PM. Tracking: 1234567890",
            "label": "legitimate"
        },
        {
            "SMS": "Reminder: Your dentist appointment is tomorrow at 10:00 AM. Call 555-1234 to reschedule.",
            "label": "legitimate"
        },
        
        # Spam examples
        {
            "SMS": "Get 50% off all items! Limited time offer. Shop now at our store!",
            "label": "spam"
        },
        {
            "SMS": "Download our new app and get $5 credit! Available on App Store and Google Play.",
            "label": "spam"
        }
    ]
    
    # Create individual JSON files
    for i, message in enumerate(sample_messages):
        filename = f"sms_message_{i+1:02d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(message, f, indent=2, ensure_ascii=False)
        
        print(f"Created: {filename}")
    
    # Create a combined JSON file with all messages
    combined_file = os.path.join(output_dir, "all_messages.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(sample_messages, f, indent=2, ensure_ascii=False)
    
    print(f"Created: all_messages.json")
    
    # Create a file with just the SMS texts (no labels)
    sms_only = [{"SMS": msg["SMS"]} for msg in sample_messages]
    sms_only_file = os.path.join(output_dir, "sms_only.json")
    with open(sms_only_file, 'w', encoding='utf-8') as f:
        json.dump(sms_only, f, indent=2, ensure_ascii=False)
    
    print(f"Created: sms_only.json")
    
    print(f"\nSample JSON files created in: {output_dir}")
    print(f"Total files created: {len(sample_messages) + 2}")


if __name__ == "__main__":
    create_sample_json_files()



