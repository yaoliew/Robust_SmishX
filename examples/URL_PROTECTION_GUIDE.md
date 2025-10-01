# URL Protection for TextAttack

A modular system for protecting URLs during both attacks and augmentations in TextAttack.

## Overview

This system provides multiple approaches to protect URLs from modification during text processing:

1. **Wrapper Approach**: Wrap existing attacks/augmenters with URL protection
2. **Decorator Approach**: Use decorators to add URL protection to methods or classes
3. **Direct Integration**: Integrate URL protection directly into custom components

## Features

- ✅ **URL Detection**: Comprehensive URL pattern matching
- ✅ **Placeholder System**: Replace URLs with unique placeholders during processing
- ✅ **Constraint Integration**: Use placeholders as stopwords to prevent modification
- ✅ **Automatic Restoration**: Restore original URLs after processing
- ✅ **Multiple Approaches**: Wrapper, decorator, and direct integration methods
- ✅ **Attack Support**: Works with all TextAttack attack methods
- ✅ **Augmentation Support**: Works with all TextAttack augmentation methods
- ✅ **Performance Optimized**: Minimal overhead for URL protection

## Quick Start

### 1. Wrapper Approach (Recommended)

```python
from textattack.augmentation import EmbeddingAugmenter
from textattack.utils.url_protection_wrapper import URLProtectedAugmenter

# Wrap any augmenter
base_augmenter = EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=4)
url_protected_augmenter = URLProtectedAugmenter(base_augmenter)

# URLs are automatically protected
text = "Visit http://example.com/path for more information"
augmented = url_protected_augmenter.augment(text)
# Result: "Visit http://example.com/path for more info" (URL preserved)
```

### 2. Decorator Approach

```python
from textattack.utils.url_protection_decorators import url_protect_method, url_protect_class

# Method decorator
class MyAugmenter:
    @url_protect_method
    def augment(self, text):
        return [text.replace("visit", "check out")]

# Class decorator
@url_protect_class
class MyAttack:
    def attack(self, attacked_text):
        # URLs are automatically protected
        pass
```

### 3. Attack Recipes

```python
from textattack.attack_recipes.url_protected_attacks import URLProtectedTextFoolerJin2019

# Use URL-protected attack recipes
attack = URLProtectedTextFoolerJin2019.build(model_wrapper)
result = attack.attack(attacked_text)
```

## API Reference

### URL Protection Wrappers

#### `URLProtectedAugmenter`

Wrapper for augmenters that protects URLs during augmentation.

```python
from textattack.utils.url_protection_wrapper import URLProtectedAugmenter

augmenter = URLProtectedAugmenter(base_augmenter)
augmented = augmenter.augment(text)
```

**Methods:**
- `augment(text)`: Augment text while protecting URLs
- `enable_url_protection()`: Enable URL protection
- `disable_url_protection()`: Disable URL protection
- `is_url_protection_enabled()`: Check if URL protection is enabled

#### `URLProtectedAttack`

Wrapper for attacks that protects URLs during the attack process.

```python
from textattack.utils.url_protection_wrapper import URLProtectedAttack

attack = URLProtectedAttack(base_attack)
result = attack.attack(attacked_text)
```

**Methods:**
- `attack(attacked_text)`: Attack text while protecting URLs
- `enable_url_protection()`: Enable URL protection
- `disable_url_protection()`: Disable URL protection
- `is_url_protection_enabled()`: Check if URL protection is enabled

### URL Protection Decorators

#### `@url_protect_method`

Decorator for individual methods that process text.

```python
from textattack.utils.url_protection_decorators import url_protect_method

class MyClass:
    @url_protect_method
    def process_text(self, text):
        return modified_text
```

#### `@url_protect_class`

Decorator for entire classes that process text.

```python
from textattack.utils.url_protection_decorators import url_protect_class

@url_protect_class
class MyAugmenter:
    def augment(self, text):
        return [modified_text]
```

#### `@url_protect_function`

Decorator for standalone functions that process text.

```python
from textattack.utils.url_protection_decorators import url_protect_function

@url_protect_function
def my_text_function(text):
    return modified_text
```

### URL Protection Utilities

#### `replace_urls_with_placeholders(text)`

Replace URLs in text with unique placeholders.

```python
from textattack.utils.url_protection import replace_urls_with_placeholders

protected_text, url_to_placeholder, placeholder_to_url = replace_urls_with_placeholders(text)
```

#### `restore_urls_from_placeholders(text, placeholder_to_url)`

Restore URLs from placeholders.

```python
from textattack.utils.url_protection import restore_urls_from_placeholders

restored_text = restore_urls_from_placeholders(protected_text, placeholder_to_url)
```

## Usage Examples

### Basic Augmentation with URL Protection

```python
from textattack.augmentation import EmbeddingAugmenter
from textattack.utils.url_protection_wrapper import URLProtectedAugmenter

# Create URL-protected augmenter
augmenter = URLProtectedAugmenter(
    EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=4)
)

# Augment text with URLs
text = "Visit http://example.com/path for more information about our services"
augmented = augmenter.augment(text)

print(f"Original: {text}")
print(f"Augmented: {augmented[0]}")
# Output: URLs are preserved exactly
```

### Attack with URL Protection

```python
from textattack.attack_recipes.url_protected_attacks import URLProtectedTextFoolerJin2019
from textattack.shared import AttackedText

# Create URL-protected attack
attack = URLProtectedTextFoolerJin2019.build(model_wrapper)

# Attack text with URLs
attacked_text = AttackedText("Visit http://example.com/path for more information")
result = attack.attack(attacked_text)

print(f"Original: {attacked_text}")
print(f"Attacked: {result.perturbed_text}")
# Output: URLs are preserved in the attack result
```

### Custom Class with URL Protection

```python
from textattack.utils.url_protection_decorators import url_protect_class

@url_protect_class
class MyCustomAugmenter:
    def __init__(self):
        self._url_protection_enabled = True
    
    def augment(self, text):
        # Your custom augmentation logic
        return [text.replace("visit", "check out")]
    
    def enable_url_protection(self):
        self._url_protection_enabled = True
    
    def disable_url_protection(self):
        self._url_protection_enabled = False

# Use the augmenter
augmenter = MyCustomAugmenter()
text = "Visit http://example.com/path for more information"
augmented = augmenter.augment(text)
# URLs are automatically protected
```

### Batch Processing with URL Protection

```python
import pandas as pd
from textattack.utils.url_protection_wrapper import URLProtectedAugmenter
from textattack.augmentation import EmbeddingAugmenter

def process_dataset_with_url_protection(input_file, output_file):
    # Read dataset
    df = pd.read_csv(input_file)
    
    # Create URL-protected augmenter
    augmenter = URLProtectedAugmenter(
        EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=2)
    )
    
    # Process each text
    augmented_data = []
    for idx, row in df.iterrows():
        text = str(row['SMS'])
        augmented_texts = augmenter.augment(text)
        
        for aug_text in augmented_texts:
            new_row = row.copy()
            new_row['SMS'] = aug_text
            augmented_data.append(new_row)
    
    # Save results
    output_df = pd.DataFrame(augmented_data)
    output_df.to_csv(output_file, index=False)
    
    print(f"Processed {len(df)} original texts into {len(output_df)} augmented texts")
    print("All URLs preserved during augmentation")

# Usage
process_dataset_with_url_protection('input.csv', 'output.csv')
```

## Supported URL Patterns

The system detects and protects various URL formats:

- `http://example.com/path`
- `https://www.example.com/path?query=value`
- `www.example.com/path`
- `example.com/path`
- `subdomain.example.com:8080/path`
- `example.com:8080/path`

## Performance Considerations

- **Minimal Overhead**: URL protection adds minimal processing time
- **Memory Efficient**: Placeholders are lightweight and temporary
- **Scalable**: Works efficiently with large datasets
- **Caching**: URL detection results can be cached for repeated processing

## Error Handling

The system gracefully handles:

- **No URLs Found**: Processes text normally without protection
- **Invalid URLs**: Skips malformed URLs and continues processing
- **Memory Issues**: Efficient placeholder management prevents memory leaks
- **Restoration Failures**: Falls back to original text if restoration fails

## Integration with Existing Code

### Minimal Changes Required

```python
# Before
augmenter = EmbeddingAugmenter(pct_words_to_swap=0.1)

# After (just wrap it)
from textattack.utils.url_protection_wrapper import URLProtectedAugmenter
augmenter = URLProtectedAugmenter(EmbeddingAugmenter(pct_words_to_swap=0.1))
```

### Backward Compatibility

- All existing TextAttack functionality remains unchanged
- URL protection is opt-in and can be disabled
- No breaking changes to existing APIs

## Troubleshooting

### Common Issues

1. **URLs Still Being Modified**
   - Ensure URL protection is enabled
   - Check that the augmenter/attack is properly wrapped
   - Verify that constraints are being updated correctly

2. **Performance Issues**
   - Consider disabling URL protection for texts without URLs
   - Use batch processing for large datasets
   - Monitor memory usage with very large texts

3. **Import Errors**
   - Ensure all required modules are installed
   - Check Python path includes TextAttack directory
   - Verify TextAttack version compatibility

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check URL protection status
print(f"URL protection enabled: {augmenter.is_url_protection_enabled()}")
```

## Contributing

To extend the URL protection system:

1. **Add New URL Patterns**: Modify `url_protection.py` to detect additional URL formats
2. **Custom Constraints**: Create custom constraints that work with URL placeholders
3. **New Decorators**: Add specialized decorators for specific use cases
4. **Performance Optimizations**: Improve URL detection and placeholder management

## License

This URL protection system is part of the TextAttack project and follows the same license terms.

