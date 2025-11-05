from typing import List, Tuple


def create_sample_sms_dataset(num_samples: int = 20) -> Tuple[List[str], List[int]]:
    legitimate_sms = [
        "Your package has been delivered. Tracking number: 1234567890",
        "Your appointment is scheduled for tomorrow at 2 PM. Please arrive 10 minutes early.",
        "Your OTP for login is 123456. Valid for 5 minutes.",
        "Your account balance is $1,234.56. Last transaction: $50.00",
        "Your flight AA1234 is delayed by 30 minutes. New departure time: 3:45 PM",
        "Your subscription will renew on 2024-01-15. Amount: $9.99",
        "Your order #ORD123 has been shipped. Expected delivery: 2-3 business days",
        "Your insurance claim has been processed. Check your email for details.",
        "Your gym membership expires in 7 days. Renew now to avoid interruption.",
        "Your credit card payment of $150.00 was successful. Thank you!",
    ]

    phishing_sms = [
        "URGENT: Your account will be suspended! Click here to verify: bit.ly/suspicious123",
        "Congratulations! You've won $1000! Claim your prize now: fake-prize.com",
        "Your bank account has been compromised. Verify immediately: bank-security-alert.net",
        "FREE iPhone 14! Limited time offer. Click to claim: free-phone-scam.org",
        "Your Netflix account expired. Update payment info: netflix-renewal-fake.com",
        "You have 3 unread messages. View now: suspicious-messages.net",
        "Your Amazon order failed. Update payment: amazon-payment-fake.com",
        "IRS Alert: Tax refund pending. Verify identity: irs-tax-refund-scam.gov",
        "Your PayPal account needs verification. Click here: paypal-verify-scam.com",
        "Win a free vacation! Enter now: vacation-scam-trip.com",
    ]

    texts: List[str] = []
    labels: List[int] = []
    for i in range(num_samples):
        if i % 2 == 0:
            texts.append(legitimate_sms[i % len(legitimate_sms)])
            labels.append(0)
        else:
            texts.append(phishing_sms[i % len(phishing_sms)])
            labels.append(1)
    return texts, labels






