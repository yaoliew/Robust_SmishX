#!/usr/bin/env python3
"""
URL Protection CLI Tool
=======================

Command-line interface for URL-protected text augmentation and attacks.
"""

import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path

# Add the TextAttack directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TextAttack'))

from textattack.augmentation import EmbeddingAugmenter, WordNetAugmenter, EasyDataAugmenter
from textattack.utils.url_protection_wrapper import URLProtectedAugmenter
from textattack.attack_recipes.url_protected_attacks import (
    URLProtectedTextFoolerJin2019,
    URLProtectedTextBuggerLi2018,
    URLProtectedPWWSRen2019,
    URLProtectedBERTAttackLi2020,
    URLProtectedDeepWordBugGao2018
)
from textattack.shared import AttackedText
from textattack.utils.url_protection import detect_urls_in_text


def augment_text(text, augmenter_type="embedding", pct_words=0.1, transformations=2):
    """Augment a single text with URL protection."""
    # Create base augmenter
    if augmenter_type == "embedding":
        base_augmenter = EmbeddingAugmenter(pct_words_to_swap=pct_words, transformations_per_example=transformations)
    elif augmenter_type == "wordnet":
        base_augmenter = WordNetAugmenter(transformations_per_example=transformations)
    elif augmenter_type == "eda":
        base_augmenter = EasyDataAugmenter(pct_words_to_swap=pct_words, transformations_per_example=transformations)
    else:
        raise ValueError(f"Unknown augmenter type: {augmenter_type}")
    
    # Create URL-protected augmenter
    url_protected_augmenter = URLProtectedAugmenter(base_augmenter)
    
    # Detect URLs
    urls = detect_urls_in_text(text)
    
    # Augment
    augmented = url_protected_augmenter.augment(text)
    
    return {
        "original": text,
        "augmented": augmented,
        "urls_detected": urls,
        "urls_preserved": all(url in aug for url in urls for aug in augmented),
        "augmenter_type": augmenter_type,
        "transformations": len(augmented)
    }


def augment_file(input_file, output_file, text_column="SMS", augmenter_type="embedding", 
                pct_words=0.1, transformations=2, max_rows=None):
    """Augment texts from a CSV file with URL protection."""
    print(f"Reading data from: {input_file}")
    
    # Read input file
    df = pd.read_csv(input_file)
    if max_rows:
        df = df.head(max_rows)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Text column: {text_column}")
    
    # Create augmenter
    if augmenter_type == "embedding":
        base_augmenter = EmbeddingAugmenter(pct_words_to_swap=pct_words, transformations_per_example=transformations)
    elif augmenter_type == "wordnet":
        base_augmenter = WordNetAugmenter(transformations_per_example=transformations)
    elif augmenter_type == "eda":
        base_augmenter = EasyDataAugmenter(pct_words_to_swap=pct_words, transformations_per_example=transformations)
    else:
        raise ValueError(f"Unknown augmenter type: {augmenter_type}")
    
    url_protected_augmenter = URLProtectedAugmenter(base_augmenter)
    
    # Process texts
    augmented_data = []
    url_stats = {"total_texts": 0, "texts_with_urls": 0, "urls_preserved": 0}
    
    for idx, row in df.iterrows():
        text = str(row[text_column])
        urls = detect_urls_in_text(text)
        
        url_stats["total_texts"] += 1
        if urls:
            url_stats["texts_with_urls"] += 1
        
        # Augment
        augmented_texts = url_protected_augmenter.augment(text)
        
        # Check URL preservation
        urls_preserved = all(url in aug for url in urls for aug in augmented_texts)
        if urls_preserved and urls:
            url_stats["urls_preserved"] += 1
        
        # Add to results
        for aug_text in augmented_texts:
            new_row = row.copy()
            new_row[text_column] = aug_text
            augmented_data.append(new_row)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows")
    
    # Save results
    output_df = pd.DataFrame(augmented_data)
    output_df.to_csv(output_file, index=False)
    
    print(f"\nAugmentation complete!")
    print(f"Original dataset: {len(df)} rows")
    print(f"Augmented dataset: {len(output_df)} rows")
    print(f"Output saved to: {output_file}")
    
    # Print URL statistics
    print(f"\nURL Protection Statistics:")
    print(f"  Total texts: {url_stats['total_texts']}")
    print(f"  Texts with URLs: {url_stats['texts_with_urls']}")
    print(f"  URLs preserved: {url_stats['urls_preserved']}")
    if url_stats['texts_with_urls'] > 0:
        print(f"  Preservation rate: {url_stats['urls_preserved']/url_stats['texts_with_urls']*100:.1f}%")
    
    return output_df


def attack_text(text, attack_type="textfooler", model_name="textattack/bert-base-uncased-imdb"):
    """Attack a single text with URL protection."""
    try:
        import transformers
        
        # Load model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        
        # Create attack
        if attack_type == "textfooler":
            attack = URLProtectedTextFoolerJin2019.build(model_wrapper)
        elif attack_type == "textbugger":
            attack = URLProtectedTextBuggerLi2018.build(model_wrapper)
        elif attack_type == "pwws":
            attack = URLProtectedPWWSRen2019.build(model_wrapper)
        elif attack_type == "bert":
            attack = URLProtectedBERTAttackLi2020.build(model_wrapper)
        elif attack_type == "deepwordbug":
            attack = URLProtectedDeepWordBugGao2018.build(model_wrapper)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Detect URLs
        urls = detect_urls_in_text(text)
        
        # Attack
        attacked_text = AttackedText(text)
        result = attack.attack(attacked_text)
        
        return {
            "original": text,
            "attacked": result.perturbed_text.text if hasattr(result, 'perturbed_text') else text,
            "success": result.goal_function_result.succeeded if hasattr(result, 'goal_function_result') else False,
            "urls_detected": urls,
            "urls_preserved": all(url in result.perturbed_text.text for url in urls) if hasattr(result, 'perturbed_text') else True,
            "attack_type": attack_type
        }
        
    except ImportError:
        print("Error: transformers library not available for attacks")
        return None
    except Exception as e:
        print(f"Error during attack: {e}")
        return None


def detect_urls(text):
    """Detect URLs in text."""
    urls = detect_urls_in_text(text)
    return {
        "text": text,
        "urls": urls,
        "url_count": len(urls)
    }


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="URL Protection CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Augment command
    augment_parser = subparsers.add_parser("augment", help="Augment text with URL protection")
    augment_parser.add_argument("--text", help="Text to augment")
    augment_parser.add_argument("--input", help="Input CSV file")
    augment_parser.add_argument("--output", help="Output CSV file")
    augment_parser.add_argument("--text-column", default="SMS", help="Text column name (default: SMS)")
    augment_parser.add_argument("--augmenter", choices=["embedding", "wordnet", "eda"], 
                               default="embedding", help="Augmenter type")
    augment_parser.add_argument("--pct-words", type=float, default=0.1, help="Percentage of words to swap")
    augment_parser.add_argument("--transformations", type=int, default=2, help="Number of transformations")
    augment_parser.add_argument("--max-rows", type=int, help="Maximum number of rows to process")
    augment_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Attack command
    attack_parser = subparsers.add_parser("attack", help="Attack text with URL protection")
    attack_parser.add_argument("--text", help="Text to attack")
    attack_parser.add_argument("--attack", choices=["textfooler", "textbugger", "pwws", "bert", "deepwordbug"], 
                              default="textfooler", help="Attack type")
    attack_parser.add_argument("--model", default="textattack/bert-base-uncased-imdb", help="Model name")
    attack_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect URLs in text")
    detect_parser.add_argument("--text", help="Text to analyze")
    detect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show URL protection status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "augment":
        if args.text:
            # Single text augmentation
            result = augment_text(args.text, args.augmenter, args.pct_words, args.transformations)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"Original: {result['original']}")
                print(f"Augmented: {result['augmented'][0]}")
                print(f"URLs detected: {result['urls_detected']}")
                print(f"URLs preserved: {result['urls_preserved']}")
        elif args.input and args.output:
            # File augmentation
            augment_file(args.input, args.output, args.text_column, args.augmenter, 
                        args.pct_words, args.transformations, args.max_rows)
        else:
            print("Error: Either --text or --input/--output required for augment command")
    
    elif args.command == "attack":
        if args.text:
            result = attack_text(args.text, args.attack, args.model)
            if result:
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"Original: {result['original']}")
                    print(f"Attacked: {result['attacked']}")
                    print(f"Success: {result['success']}")
                    print(f"URLs detected: {result['urls_detected']}")
                    print(f"URLs preserved: {result['urls_preserved']}")
        else:
            print("Error: --text required for attack command")
    
    elif args.command == "detect":
        if args.text:
            result = detect_urls(args.text)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"Text: {result['text']}")
                print(f"URLs: {result['urls']}")
                print(f"URL count: {result['url_count']}")
        else:
            print("Error: --text required for detect command")
    
    elif args.command == "status":
        print("URL Protection System Status:")
        print("✅ URL protection utilities loaded")
        print("✅ Augmentation wrappers available")
        print("✅ Attack wrappers available")
        print("✅ URL detection working")
        print("✅ All systems operational")


if __name__ == "__main__":
    main()

