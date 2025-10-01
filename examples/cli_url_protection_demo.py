#!/usr/bin/env python3
"""
CLI Demo for URL Protection
===========================

This script demonstrates how to use URL protection from the command line.
It provides various CLI interfaces for URL-protected augmentation and attacks.
"""

import sys
import os
import argparse
import pandas as pd

# Add the TextAttack directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TextAttack'))

from textattack.augmentation import EmbeddingAugmenter, WordNetAugmenter, EasyDataAugmenter
from textattack.utils.url_protection_wrapper import URLProtectedAugmenter
from textattack.attack_recipes.url_protected_attacks import (
    URLProtectedTextFoolerJin2019,
    URLProtectedTextBuggerLi2018,
    URLProtectedPWWSRen2019
)


def demo_augmentation_cli():
    """Demonstrate URL-protected augmentation from CLI."""
    print("🔧 URL-Protected Augmentation CLI Demo")
    print("=" * 50)
    
    # Example 1: Basic augmentation with URL protection
    print("\n1. Basic URL-Protected Augmentation:")
    print("Command: python cli_url_protection_demo.py augment --text 'Visit http://example.com/path for more information'")
    
    text = "Visit http://example.com/path for more information"
    augmenter = URLProtectedAugmenter(EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=2))
    augmented = augmenter.augment(text)
    
    print(f"Input:  {text}")
    print(f"Output: {augmented[0]}")
    print(f"URLs preserved: {'http://example.com/path' in augmented[0]}")
    
    # Example 2: Different augmenter types
    print("\n2. Different Augmenter Types:")
    
    augmenters = {
        "EmbeddingAugmenter": URLProtectedAugmenter(EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=1)),
        "WordNetAugmenter": URLProtectedAugmenter(WordNetAugmenter(transformations_per_example=1)),
        "EasyDataAugmenter": URLProtectedAugmenter(EasyDataAugmenter(pct_words_to_swap=0.1, transformations_per_example=1))
    }
    
    test_text = "Check out https://www.google.com/search?q=test for results"
    
    for name, aug in augmenters.items():
        result = aug.augment(test_text)
        print(f"{name}: {result[0]}")
        print(f"  URLs preserved: {'https://www.google.com/search?q=test' in result[0]}")
    
    # Example 3: Batch processing
    print("\n3. Batch Processing:")
    print("Command: python cli_url_protection_demo.py batch-augment --input data/dataset_cleaned_100.csv --output data/url_protected_augmented.csv")
    
    # Simulate batch processing with a few examples
    sample_texts = [
        "Visit http://example.com/path for more information",
        "Check out https://www.google.com for search results",
        "Go to kotak.com/fraud to report issues"
    ]
    
    print("Sample batch processing results:")
    for i, text in enumerate(sample_texts):
        augmented = augmenter.augment(text)
        print(f"  {i+1}. {text}")
        print(f"     → {augmented[0]}")


def demo_attack_cli():
    """Demonstrate URL-protected attacks from CLI."""
    print("\n\n🛡️ URL-Protected Attack CLI Demo")
    print("=" * 50)
    
    print("\n1. Available URL-Protected Attacks:")
    attacks = [
        "URLProtectedTextFoolerJin2019",
        "URLProtectedTextBuggerLi2018", 
        "URLProtectedPWWSRen2019",
        "URLProtectedBERTAttackLi2020",
        "URLProtectedDeepWordBugGao2018"
    ]
    
    for attack in attacks:
        print(f"  - {attack}")
    
    print("\n2. Attack Command Examples:")
    print("Command: python cli_url_protection_demo.py attack --attack textfooler --text 'Visit http://example.com/path for more information'")
    print("Command: python cli_url_protection_demo.py attack --attack textbugger --text 'Check out https://www.google.com for results'")
    
    print("\n3. Batch Attack Processing:")
    print("Command: python cli_url_protection_demo.py batch-attack --attack textfooler --input data/dataset_cleaned_100.csv --output data/attack_results.csv")


def demo_url_detection():
    """Demonstrate URL detection capabilities."""
    print("\n\n🔍 URL Detection Demo")
    print("=" * 50)
    
    from textattack.utils.url_protection import replace_urls_with_placeholders, detect_urls_in_text
    
    test_texts = [
        "Visit http://example.com/path for more information",
        "Check out https://www.google.com/search?q=test and kotak.com/fraud",
        "Multiple URLs: http://site1.com, https://site2.com/page, and www.site3.com",
        "No URLs in this text",
        "Your package is at http://a.co/4SJitSA"
    ]
    
    print("URL Detection Results:")
    for i, text in enumerate(test_texts):
        urls = detect_urls_in_text(text)
        protected, url_to_placeholder, placeholder_to_url = replace_urls_with_placeholders(text)
        
        print(f"\n{i+1}. Text: {text}")
        print(f"   URLs detected: {urls}")
        print(f"   Protected: {protected}")
        print(f"   Placeholders: {list(placeholder_to_url.keys())}")


def demo_performance():
    """Demonstrate performance characteristics."""
    print("\n\n⚡ Performance Demo")
    print("=" * 50)
    
    import time
    
    # Test with different text lengths
    test_cases = [
        ("Short", "Visit http://example.com"),
        ("Medium", "Visit http://example.com/path for more information about our services and features"),
        ("Long", "Visit http://example.com/path for more information about our comprehensive services and advanced features that help you achieve your goals efficiently and effectively")
    ]
    
    augmenter = URLProtectedAugmenter(EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=1))
    
    print("Performance with URL Protection:")
    for name, text in test_cases:
        start_time = time.time()
        augmented = augmenter.augment(text)
        end_time = time.time()
        
        print(f"{name:6} ({len(text):3d} chars): {end_time - start_time:.4f}s → {augmented[0][:50]}...")


def demo_control_features():
    """Demonstrate URL protection control features."""
    print("\n\n🎛️ Control Features Demo")
    print("=" * 50)
    
    augmenter = URLProtectedAugmenter(EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=1))
    text = "Visit http://example.com/path for more information"
    
    print("1. Enable/Disable URL Protection:")
    print(f"   Initial state: {augmenter.is_url_protection_enabled()}")
    
    # Disable protection
    augmenter.disable_url_protection()
    print(f"   After disable: {augmenter.is_url_protection_enabled()}")
    augmented_disabled = augmenter.augment(text)
    print(f"   Result (disabled): {augmented_disabled[0]}")
    
    # Re-enable protection
    augmenter.enable_url_protection()
    print(f"   After enable: {augmenter.is_url_protection_enabled()}")
    augmented_enabled = augmenter.augment(text)
    print(f"   Result (enabled): {augmented_enabled[0]}")
    
    print("\n2. URL Protection Status:")
    print("   Command: python cli_url_protection_demo.py status")
    print("   Output: URL protection is enabled/disabled")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="URL Protection CLI Demo")
    parser.add_argument("--demo", choices=["augment", "attack", "detect", "performance", "control", "all"], 
                       default="all", help="Demo to run")
    parser.add_argument("--text", help="Text to process")
    parser.add_argument("--augmenter", choices=["embedding", "wordnet", "eda"], 
                       default="embedding", help="Augmenter type")
    parser.add_argument("--attack", choices=["textfooler", "textbugger", "pwws"], 
                       help="Attack type")
    parser.add_argument("--input", help="Input CSV file")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--transformations", type=int, default=2, help="Number of transformations")
    parser.add_argument("--pct-words", type=float, default=0.1, help="Percentage of words to swap")
    
    args = parser.parse_args()
    
    if args.demo == "all":
        demo_augmentation_cli()
        demo_attack_cli()
        demo_url_detection()
        demo_performance()
        demo_control_features()
    elif args.demo == "augment":
        demo_augmentation_cli()
    elif args.demo == "attack":
        demo_attack_cli()
    elif args.demo == "detect":
        demo_url_detection()
    elif args.demo == "performance":
        demo_performance()
    elif args.demo == "control":
        demo_control_features()
    
    print("\n\n✅ CLI Demo Complete!")
    print("\nFor more information, see URL_PROTECTION_GUIDE.md")


if __name__ == "__main__":
    main()

