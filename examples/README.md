# URL Protection Examples and Documentation

This folder contains examples, documentation, and demonstration scripts for the URL protection system.

## Files

### `URL_PROTECTION_GUIDE.md`
Comprehensive guide covering:
- Overview of the URL protection system
- Multiple implementation approaches (wrapper, decorator, direct integration)
- API reference for all components
- Usage examples for attacks and augmentations
- Performance considerations and troubleshooting

### `CLI_USAGE_EXAMPLES.md`
Detailed CLI usage examples including:
- Basic command-line operations
- Batch processing workflows
- Different augmenter types
- JSON output formatting
- Integration with existing pipelines
- Error handling and troubleshooting

### `cli_url_protection_demo.py`
Interactive demonstration script showing:
- URL-protected augmentation examples
- Attack method demonstrations
- URL detection capabilities
- Performance characteristics
- Control features (enable/disable protection)

## Quick Start

### 1. Run the Demo
```bash
python examples/cli_url_protection_demo.py --demo all
```

### 2. Try CLI Commands
```bash
# Basic augmentation
python util/url_protect_cli.py augment --text "Visit http://example.com/path for more information"

# Batch processing
python util/url_protect_cli.py augment --input data/dataset_cleaned_100.csv --output data/augmented.csv

# URL detection
python util/url_protect_cli.py detect --text "Check out https://www.google.com for results"
```

### 3. Read the Documentation
- Start with `URL_PROTECTION_GUIDE.md` for comprehensive understanding
- Use `CLI_USAGE_EXAMPLES.md` for command-line reference

## Demo Types

The demo script supports different demonstration types:

```bash
# All demonstrations
python examples/cli_url_protection_demo.py --demo all

# Specific demonstrations
python examples/cli_url_protection_demo.py --demo augment    # Augmentation examples
python examples/cli_url_protection_demo.py --demo attack     # Attack examples
python examples/cli_url_protection_demo.py --demo detect     # URL detection
python examples/cli_url_protection_demo.py --demo performance # Performance tests
python examples/cli_url_protection_demo.py --demo control    # Control features
```

## Integration Examples

### Python Integration
```python
# Import from TextAttack
from textattack.augmentation import URLProtectedEmbeddingAugmenter

# Use the augmenter
augmenter = URLProtectedEmbeddingAugmenter()
result = augmenter.augment("Visit http://example.com for more info")
```

### CLI Integration
```bash
# Process your dataset
python util/url_protect_cli.py augment \
    --input your_data.csv \
    --output augmented_data.csv \
    --text-column "text" \
    --augmenter embedding \
    --transformations 2
```

## File Structure

```
examples/
├── README.md                    # This file
├── URL_PROTECTION_GUIDE.md     # Comprehensive documentation
├── CLI_USAGE_EXAMPLES.md       # CLI usage examples
└── cli_url_protection_demo.py  # Interactive demo script
```

## Support

For questions or issues:
1. Check the documentation in this folder
2. Run the demo script to see examples
3. Review the utility files in the `util/` folder
4. Examine the core implementation in `TextAttack/textattack/utils/`
