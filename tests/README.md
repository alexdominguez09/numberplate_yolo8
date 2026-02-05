# Testing the Number Plate Recognition System

This directory contains comprehensive unit and integration tests for the number plate recognition system using YOLOv8.

## Test Structure

### Unit Tests (`test_utils.py`)
Tests for individual utility functions in `utils.py`:

1. **`check_license_plate_format(text)`** - Validates UK license plate format
   - Valid plates (AB12CDE, A123BCD, etc.)
   - Invalid length, character positions, edge cases

2. **`format_license_number(text)`** - Formats ambiguous characters
   - 0 → O, 1 → I, 3 → J, 4 → A, 5 → S, 6 → G mappings
   - Position-specific mapping logic

3. **`read_license_plate(img)`** - OCR text extraction with EasyOCR
   - Valid OCR detection
   - Invalid format handling
   - Multiple detections
   - No detection cases

4. **`map_car(plate, tracking_ids)`** - Associates plates with vehicles
   - Plate inside/outside vehicle bounding boxes
   - Multiple vehicles
   - Empty tracking IDs

5. **`write_csv(results, output_path)`** - CSV output generation
   - Basic results writing
   - Empty results
   - Missing keys handling
   - File permission errors

6. **Dictionary consistency tests**
   - `dict_char_to_int` and `dict_int_to_char` mappings
   - Consistency between dictionaries

### Integration Tests

**Simple Integration Tests (`test_simple_integration.py`)**
- CSV format compatibility between components
- Bounding box parsing consistency
- Vehicle filtering logic
- Plate-to-vehicle mapping
- License plate validation

**Complex Integration Tests (`test_integration.py`)**
- Full pipeline with mocked components
- Data flow through main pipeline
- Interpolation pipeline
- Visualization pipeline
- End-to-end format compatibility

### Test Fixtures (`conftest.py`)
Common test fixtures:
- Temporary CSV files
- Sample license plate images
- Vehicle tracking IDs
- Plate detections
- Detection results
- Mocked components (OCR, video capture, YOLO, SORT)

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install -r requirements-dev.txt
```

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/test_utils.py -v

# Simple integration tests
python -m pytest tests/test_simple_integration.py -v

# With coverage reporting
python -m pytest tests/ --cov=utils --cov-report=html
```

### Test Coverage
The tests cover:
- **Core utility functions**: 100% of `utils.py` logic
- **Edge cases**: Invalid inputs, boundary conditions
- **Integration points**: CSV format compatibility, data flow
- **Mocking**: External dependencies (OCR, YOLO, video I/O)

## Key Test Findings

### 1. License Plate Format Validation
- UK plates must be 7 characters
- Positions 0-1: letters or mappable numbers (0/O, 1/I)
- Positions 2-3: digits or mappable letters (O/0, I/1, J/3, A/4, G/6, S/5)
- Positions 4-6: letters or mappable numbers

### 2. Character Mapping Logic
- `format_license_number()` uses position-specific mapping:
  - Positions 0,1,4,5,6: `dict_int_to_char` (digits → letters)
  - Positions 2,3: `dict_char_to_int` (letters → digits)

### 3. CSV Format Details
- `write_csv()` adds spaces after commas: `"frame_nmb, car_id, ..."`
- Bounding boxes formatted as: `" [x1 y1 x2 y2]"` (space before bracket)
- `interpolate_data.py` parses with `[2:-1]` to remove `' ['` and `']'`
- `visualize_data.py` parses with `ast.literal_eval(bbox.replace(' ', ','))`

### 4. Potential Issues Found
1. `read_license_plate()` returns after first OCR detection (doesn't check all)
2. `map_car()` uses strict inequality: plate must be fully inside vehicle
3. `format_license_number()` expects exactly 7 characters (IndexError otherwise)

## Adding New Tests

### For New Utility Functions
1. Create test class in `test_utils.py`
2. Test valid inputs, edge cases, error conditions
3. Use appropriate fixtures from `conftest.py`

### For Integration Tests
1. Mock external dependencies (YOLO, OCR, video I/O)
2. Test data flow between components
3. Verify format compatibility

### Best Practices
- Use `tmp_path` fixture for temporary files
- Mock expensive operations (OCR, YOLO inference)
- Test both success and failure paths
- Include descriptive test names and docstrings

## Continuous Integration

To set up CI (GitHub Actions, GitLab CI, etc.):

1. Install dependencies from `requirements-dev.txt`
2. Run tests with `pytest tests/`
3. Generate coverage reports
4. Optionally run linting/type checking

Example GitHub Actions workflow would include:
- Python 3.11 environment
- Dependency installation
- Test execution
- Coverage reporting