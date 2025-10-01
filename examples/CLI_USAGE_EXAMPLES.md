# CLI Usage Examples for URL Protection

This document demonstrates how to use the URL protection system from the command line interface.

## Available CLI Tools

### 1. `url_protect_cli.py` - Main CLI Tool
A comprehensive command-line interface for URL-protected text processing.

### 2. `cli_url_protection_demo.py` - Demo Script
A demonstration script showing various URL protection capabilities.

## Basic Usage Examples

### Text Augmentation with URL Protection

```bash
# Single text augmentation
python url_protect_cli.py augment --text "Visit http://example.com/path for more information"

# Output:
# Original: Visit http://example.com/path for more information
# Augmented: Visit http://example.com/path for more info
# URLs detected: ['http://example.com/path', 'example.com/path']
# URLs preserved: True
```

### Different Augmenter Types

```bash
# Embedding-based augmentation (default)
python url_protect_cli.py augment --text "Visit http://example.com/path" --augmenter embedding

# WordNet-based augmentation
python url_protect_cli.py augment --text "Visit http://example.com/path" --augmenter wordnet

# Easy Data Augmentation (EDA)
python url_protect_cli.py augment --text "Visit http://example.com/path" --augmenter eda
```

### Batch Processing

```bash
# Process CSV file with URL protection
python url_protect_cli.py augment \
    --input data/dataset_cleaned_100.csv \
    --output data/url_protected_augmented.csv \
    --text-column SMS \
    --transformations 2 \
    --max-rows 100

# Output includes URL protection statistics:
# URL Protection Statistics:
#   Total texts: 100
#   Texts with URLs: 45
#   URLs preserved: 45
#   Preservation rate: 100.0%
```

### URL Detection

```bash
# Detect URLs in text
python url_protect_cli.py detect --text "Check out https://www.google.com/search?q=test and kotak.com/fraud"

# Output:
# Text: Check out https://www.google.com/search?q=test and kotak.com/fraud
# URLs: ['https://www.google.com/search?q=test', 'www.google.com/search?q=test', 'kotak.com/fraud']
# URL count: 3
```

### JSON Output

```bash
# Get structured JSON output
python url_protect_cli.py augment --text "Visit http://example.com/path" --json

# Output:
# {
#   "original": "Visit http://example.com/path for more information",
#   "augmented": [
#     "Visited http://example.com/path for more information",
#     "Visits http://example.com/path for more information"
#   ],
#   "urls_detected": [
#     "http://example.com/path",
#     "example.com/path"
#   ],
#   "urls_preserved": true,
#   "augmenter_type": "embedding",
#   "transformations": 2
# }
```

### System Status

```bash
# Check URL protection system status
python url_protect_cli.py status

# Output:
# URL Protection System Status:
# ✅ URL protection utilities loaded
# ✅ Augmentation wrappers available
# ✅ Attack wrappers available
# ✅ URL detection working
# ✅ All systems operational
```

## Advanced Usage

### Custom Parameters

```bash
# Custom augmentation parameters
python url_protect_cli.py augment \
    --text "Visit http://example.com/path for more information" \
    --augmenter embedding \
    --pct-words 0.2 \
    --transformations 4

# Process specific number of rows
python url_protect_cli.py augment \
    --input data/large_dataset.csv \
    --output data/sample_augmented.csv \
    --max-rows 50 \
    --transformations 1
```

### Different Text Columns

```bash
# Process CSV with different text column name
python url_protect_cli.py augment \
    --input data/messages.csv \
    --output data/augmented_messages.csv \
    --text-column "message_text" \
    --augmenter wordnet
```

## Demo Script Usage

### Run All Demos

```bash
python cli_url_protection_demo.py --demo all
```

### Specific Demo Types

```bash
# Augmentation demo only
python cli_url_protection_demo.py --demo augment

# Attack demo only
python cli_url_protection_demo.py --demo attack

# URL detection demo only
python cli_url_protection_demo.py --demo detect

# Performance demo only
python cli_url_protection_demo.py --demo performance

# Control features demo only
python cli_url_protection_demo.py --demo control
```

## Integration with Existing Workflows

### Using with TextAttack CLI

```bash
# Traditional TextAttack augmentation
textattack augment --input-csv data/input.csv --output-csv data/output.csv

# URL-protected augmentation (using our CLI)
python url_protect_cli.py augment --input data/input.csv --output data/url_protected_output.csv
```

### Pipeline Integration

```bash
# Example pipeline: detect URLs, then augment with protection
python url_protect_cli.py detect --text "$(cat input.txt)" --json > urls.json
python url_protect_cli.py augment --input input.csv --output augmented.csv
```

### Batch Processing Script

```bash
#!/bin/bash
# Process multiple files with URL protection

for file in data/*.csv; do
    output_file="augmented_$(basename "$file")"
    echo "Processing $file -> $output_file"
    python url_protect_cli.py augment \
        --input "$file" \
        --output "$output_file" \
        --transformations 2
done
```

## Error Handling

### Common Issues and Solutions

```bash
# File not found
python url_protect_cli.py augment --input nonexistent.csv
# Error: File not found

# Invalid augmenter type
python url_protect_cli.py augment --text "test" --augmenter invalid
# Error: Unknown augmenter type: invalid

# Missing required arguments
python url_protect_cli.py augment
# Error: Either --text or --input/--output required for augment command
```

## Performance Tips

### For Large Datasets

```bash
# Process in smaller batches
python url_protect_cli.py augment \
    --input large_dataset.csv \
    --output batch1.csv \
    --max-rows 1000

# Use fewer transformations for speed
python url_protect_cli.py augment \
    --input data.csv \
    --output output.csv \
    --transformations 1 \
    --pct-words 0.05
```

### Memory Optimization

```bash
# Process with minimal memory usage
python url_protect_cli.py augment \
    --input data.csv \
    --output output.csv \
    --max-rows 100 \
    --transformations 1
```

## Output Formats

### CSV Output
- Preserves all original columns
- Adds augmented text in the same text column
- Maintains data integrity

### JSON Output
- Structured data with metadata
- Includes URL detection results
- Shows preservation statistics

### Console Output
- Human-readable format
- Progress indicators for batch processing
- Error messages and warnings

## Troubleshooting

### Check System Status
```bash
python url_protect_cli.py status
```

### Verify URL Detection
```bash
python url_protect_cli.py detect --text "Your test text with http://example.com"
```

### Test with Simple Text
```bash
python url_protect_cli.py augment --text "Simple text without URLs"
```

## Best Practices

1. **Always check system status** before processing large datasets
2. **Use appropriate augmenter types** for your use case
3. **Monitor URL preservation rates** in batch processing
4. **Process in batches** for large datasets to avoid memory issues
5. **Use JSON output** for programmatic integration
6. **Test with small samples** before full processing

## Examples with Real Data

### SMS Dataset Processing

```bash
# Process SMS dataset with URL protection
python url_protect_cli.py augment \
    --input data/sms_dataset.csv \
    --output data/augmented_sms.csv \
    --text-column "message" \
    --augmenter embedding \
    --transformations 2

# Check results
head -5 data/augmented_sms.csv
```

### E-commerce Text Processing

```bash
# Process product descriptions
python url_protect_cli.py augment \
    --input data/products.csv \
    --output data/augmented_products.csv \
    --text-column "description" \
    --augmenter wordnet \
    --pct-words 0.1
```

This CLI system provides a powerful and flexible interface for URL-protected text processing, making it easy to integrate URL protection into existing workflows and pipelines.

