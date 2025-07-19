# Final Project Evaluation Output Verifyer

A simple Python script that validates JSON files containing evaluation outputs against a predefined structure using Pydantic.

## Overview

This tool verifies that JSON files follow the expected structure for evaluation outputs, ensuring data consistency and catching validation errors early in your data processing pipeline.

## Expected JSON Structure

The script validates JSON files that contain an **array** of evaluation objects. Each object in the array must have the following structure:

```json
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "top_k": {
            "1": "Paris",
            "2": "London", 
            "3": "Berlin"
        }
    },
    {
        "question": "What is the capital of Germany?",
        "answer": "Berlin",
        "top_k": {
            "1": "Berlin",
            "2": "Paris",
            "3": "London"
        }
    }
]
```

### Validation Rules

- ✅ **Root element**: Must be an array
- ✅ **question**: Required string field
- ✅ **answer**: Required string field  
- ✅ **top_k**: Required dictionary with exactly 3 string key-value pairs
- ✅ **Type checking**: All fields must be of the correct data type
- ✅ **Completeness**: No missing required fields allowed

## Installation

1. **Clone or download** this repository

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install Pydantic directly:
   ```bash
   pip install pydantic>=2.0.0
   ```

## Usage

### Basic Usage

Validate a JSON file:
```bash
python3 read_evals.py sample_output.json
```

**Expected output** (on success):
```
Successfully validated 2 items
✓ 'sample_output.json' has valid structure
```

### Quiet Mode

For automated scripts or CI/CD pipelines, use quiet mode to get just the result:
```bash
python3 read_evals.py -q sample_output.json
```

**Output**: `VALID` or `INVALID`

### Exit Codes

- **0**: Validation successful
- **1**: Validation failed (useful for shell scripts and CI/CD)

## Error Examples

### Missing Required Field
```json
[
    {
        "question": "What is the capital of France?",
        "top_k": {"1": "Paris", "2": "London", "3": "Berlin"}
    }
]
```
**Error**: `Validation failed for item 1 - Field required [type=missing, input={'question': '...', 'top_k': {...}}]`

### Wrong Number of top_k Items
```json
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "top_k": {
            "1": "Paris",
            "2": "London"
        }
    }
]
```
**Error**: `Validation failed for item 1 - top_k must have exactly 3 items, got 2`

### Invalid Root Structure
```json
{
    "question": "What is the capital of France?",
    "answer": "Paris",
    "top_k": {"1": "Paris", "2": "London", "3": "Berlin"}
}
```
**Error**: `Root element must be an array, got dict`

## Command Line Options

```
usage: read_evals.py [-h] [-q] filename

Verify JSON file structure against sample_output.json using Pydantic

positional arguments:
  filename       Path to the JSON file to validate

options:
  -h, --help     show this help message and exit
  -q, --quiet    Only output result, no details
```

## Integration Examples

### Shell Script
```bash
#!/bin/bash
if python3 read_evals.py -q data.json; then
    echo "Data validation passed"
    # Continue processing...
else
    echo "Data validation failed"
    exit 1
fi
```

### Python Integration
```python
import subprocess
import sys

def validate_json_file(filename):
    """Validate JSON file and return True if valid"""
    result = subprocess.run([
        'python3', 'read_evals.py', '-q', filename
    ], capture_output=True, text=True)
    
    return result.returncode == 0

# Usage
if validate_json_file('my_data.json'):
    print("File is valid!")
else:
    print("File validation failed!")
```

## Files

- **`read_evals.py`**: Main validation script
- **`requirements.txt`**: Python dependencies
- **`sample_output.json`**: Example of correctly formatted JSON
- **`README.md`**: This documentation

## Dependencies

- **Python 3.7+**
- **Pydantic 2.0+**: For data validation and parsing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Feel free to use and modify as needed.

## Troubleshooting

### Common Issues

**Import Error**: `Import "pydantic" could not be resolved`
- **Solution**: Install dependencies with `pip install -r requirements.txt`

**Validation Errors**: Check that your JSON file matches the expected structure exactly
- Ensure it's an array at the root level
- Verify all required fields are present
- Check that `top_k` has exactly 3 items

**File Not Found**: Make sure the JSON file path is correct
- Use absolute paths if needed
- Check file permissions 