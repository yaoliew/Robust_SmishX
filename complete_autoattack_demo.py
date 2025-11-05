#!/usr/bin/env python3
"""
Complete AutoAttack Demonstration Script
This script shows the complete workflow of using AutoAttack with Qwen-VL-7B-Instruct for SMS phishing detection.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from autoattack import AutoAttack
import numpy as np
import logging
import json
import os
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_banner():
    """Print a banner for the demonstration."""
    print("="*80)
    print("AUTOATTACK INTEGRATION WITH QWEN-VL-7B-INSTRUCT")
    print("SMS Phishing Detection Robustness Evaluation")
    print("="*80)
    print("This demonstration shows how to use AutoAttack to evaluate")
    print("the adversarial robustness of SMS phishing detection models.")
    print("="*80)

class SMSPhishingRobustClassifier(nn.Module):
    """
    Robust SMS Phishing detection classifier for AutoAttack evaluation.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Robust classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification
        )
        
        # Move to device
        self.transformer = self.transformer.to(self.device)
        self.classifier = self.classifier.to(self.device)
        
        self.transformer.eval()
        self.classifier.eval()
        
        logger.info("SMS Phishing classifier loaded successfully")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for AutoAttack compatibility."""
        logits = self.classifier(x)
        return logits
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get transformer embeddings for texts."""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.transformer(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings

def create_evaluation_dataset() -> Tuple[List[str], List[int]]:
    """Create a comprehensive evaluation dataset."""
    
    legitimate_sms = [
        "Your package has been delivered. Tracking: 1Z999AA1234567890. Thank you!",
        "Appointment reminder: Dental checkup tomorrow at 2:30 PM. Arrive 10 min early.",
        "Verification code: 847392. Expires in 5 minutes. Do not share with anyone.",
        "Account balance: $1,234.56. Last transaction: $25.99 at Starbucks.",
        "Flight AA1234 delayed 45 minutes. New departure: 4:15 PM. We apologize.",
        "Monthly subscription renews Jan 20, 2024. Amount: $9.99 charged to card ending 1234.",
        "Order #ORD-2024-001 shipped via UPS. Delivery: 2-3 business days.",
        "Insurance claim #CLM-789456 processed and approved. Check email for details.",
        "Gym membership expires in 7 days. Renew now to avoid interruption.",
        "Payment successful! $150.00 charged to credit card ending 5678. Thank you.",
        "Your Uber ride is arriving in 3 minutes. Driver: John, Vehicle: Toyota Camry.",
        "Your Netflix subscription is active. Next billing date: February 15, 2024.",
        "Your Amazon Prime membership expires in 30 days. Renew to continue benefits.",
        "Your bank statement is ready. View online or download PDF from our app.",
        "Your hotel reservation is confirmed for March 15-17, 2024. Check-in: 3 PM."
    ]
    
    phishing_sms = [
        "URGENT: Bank account suspended in 24 hours! Verify: bit.ly/bank-verify-urgent",
        "Congratulations! You won $1,000 cash! Claim now: prize-winner-2024.com",
        "Security Alert: Account compromised. Click to secure: bank-security-alert.net",
        "FREE iPhone 15 Pro! Limited offer. Claim: apple-giveaway-scam.org",
        "Netflix subscription expired. Update payment: netflix-renewal-fake.com/update",
        "You have 5 unread messages. View: suspicious-messages-app.net",
        "Amazon Alert: Payment failed. Update: amazon-payment-scam.com",
        "IRS Notice: Tax refund pending verification. Verify: irs-tax-refund-scam.gov",
        "PayPal Security: Unusual activity. Verify account: paypal-security-fake.com",
        "Win free Hawaii vacation! Enter: vacation-scam-trip.com/hawaii-giveaway",
        "Your Apple ID will be locked. Verify immediately: apple-id-security-fake.com",
        "Google account suspended. Click to restore: google-account-recovery-scam.org",
        "Your Microsoft account needs verification. Verify: microsoft-security-fake.com",
        "Your WhatsApp account will be deleted. Restore: whatsapp-recovery-scam.net",
        "Your Instagram account is at risk. Secure now: instagram-security-fake.com"
    ]
    
    texts = legitimate_sms + phishing_sms
    labels = [0] * len(legitimate_sms) + [1] * len(phishing_sms)
    
    logger.info(f"Created evaluation dataset with {len(texts)} samples")
    return texts, labels

def run_complete_evaluation():
    """Run complete AutoAttack evaluation."""
    logger.info("Starting complete AutoAttack evaluation")
    
    # Load model
    model = SMSPhishingRobustClassifier()
    
    # Create dataset
    texts, labels = create_evaluation_dataset()
    
    # Get embeddings
    logger.info("Computing text embeddings...")
    embeddings = model.get_embeddings(texts)
    labels_tensor = torch.tensor(labels).to(model.device)
    
    logger.info(f"Evaluation dataset: {len(embeddings)} samples, {embeddings.shape[1]} dimensions")
    
    # Calculate clean accuracy
    with torch.no_grad():
        clean_logits = model(embeddings)
        clean_pred = clean_logits.argmax(dim=1)
        clean_acc = (clean_pred == labels_tensor).float().mean().item()
    
    logger.info(f"Clean accuracy: {clean_acc:.4f}")
    
    # Test different AutoAttack configurations
    configurations = [
        {'epsilon': 0.1, 'norm': 'Linf', 'version': 'standard', 'description': 'Small Linf attack'},
        {'epsilon': 0.2, 'norm': 'Linf', 'version': 'standard', 'description': 'Medium Linf attack'},
        {'epsilon': 0.3, 'norm': 'Linf', 'version': 'standard', 'description': 'Large Linf attack'},
    ]
    
    results = {
        'clean_accuracy': clean_acc,
        'evaluations': [],
        'summary': {}
    }
    
    for config in configurations:
        logger.info(f"Testing: {config['description']} (ε={config['epsilon']}, {config['norm']})")
        
        try:
            # Define forward pass function
            def forward_pass(x: torch.Tensor) -> torch.Tensor:
                return model(x)
            
            # Initialize AutoAttack
            adversary = AutoAttack(
                forward_pass, 
                norm=config['norm'], 
                eps=config['epsilon'], 
                version=config['version'],
                log_path=f'complete_eval_log_{config["epsilon"]}_{config["norm"]}.txt'
            )
            
            # Run evaluation
            x_adv = adversary.run_standard_evaluation(embeddings, labels_tensor, bs=2)
            
            # Calculate adversarial accuracy
            with torch.no_grad():
                adv_logits = model(x_adv)
                adv_pred = adv_logits.argmax(dim=1)
                adv_acc = (adv_pred == labels_tensor).float().mean().item()
            
            evaluation_result = {
                **config,
                'adversarial_accuracy': adv_acc,
                'robust_accuracy': adv_acc,
                'num_samples': len(embeddings),
                'success': True
            }
            
            results['evaluations'].append(evaluation_result)
            logger.info(f"Adversarial accuracy: {adv_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Configuration {config['description']} failed: {e}")
            evaluation_result = {
                **config,
                'adversarial_accuracy': 0.0,
                'robust_accuracy': 0.0,
                'num_samples': len(embeddings),
                'error': str(e),
                'success': False
            }
            results['evaluations'].append(evaluation_result)
    
    # Calculate summary statistics
    successful_evaluations = [eval for eval in results['evaluations'] if eval['success']]
    if successful_evaluations:
        adv_accuracies = [eval['adversarial_accuracy'] for eval in successful_evaluations]
        results['summary'] = {
            'min_adversarial_accuracy': min(adv_accuracies),
            'max_adversarial_accuracy': max(adv_accuracies),
            'mean_adversarial_accuracy': np.mean(adv_accuracies),
            'robustness_gap': clean_acc - np.mean(adv_accuracies),
            'successful_evaluations': len(successful_evaluations),
            'total_evaluations': len(results['evaluations'])
        }
    
    return results

def print_results(results: Dict):
    """Print comprehensive results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Clean Accuracy: {results['clean_accuracy']:.4f}")
    
    if results['summary']:
        print(f"Min Adversarial Accuracy: {results['summary']['min_adversarial_accuracy']:.4f}")
        print(f"Max Adversarial Accuracy: {results['summary']['max_adversarial_accuracy']:.4f}")
        print(f"Mean Adversarial Accuracy: {results['summary']['mean_adversarial_accuracy']:.4f}")
        print(f"Robustness Gap: {results['summary']['robustness_gap']:.4f}")
        print(f"Successful Evaluations: {results['summary']['successful_evaluations']}/{results['summary']['total_evaluations']}")
    
    print("\nDetailed Results:")
    for eval_result in results['evaluations']:
        if eval_result['success']:
            print(f"  ✓ {eval_result['description']}: {eval_result['adversarial_accuracy']:.4f}")
        else:
            print(f"  ✗ {eval_result['description']}: FAILED")
    
    print("="*80)

def print_interpretation():
    """Print interpretation of results."""
    print("\nINTERPRETATION:")
    print("- Clean accuracy: Model performance on normal inputs")
    print("- Adversarial accuracy: Model performance under attack")
    print("- Robustness gap: Performance drop under adversarial conditions")
    print("- Lower adversarial accuracy indicates higher vulnerability")
    print("- Large robustness gap suggests need for adversarial training")
    print("="*80)

def main():
    print_banner()
    
    # Run evaluation
    results = run_complete_evaluation()
    
    # Create output directory
    os.makedirs('complete_autoattack_results', exist_ok=True)
    
    # Save results
    results_file = 'complete_autoattack_results/complete_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print_results(results)
    print_interpretation()
    
    print(f"\nResults saved to: {results_file}")
    print("="*80)

if __name__ == "__main__":
    main()




