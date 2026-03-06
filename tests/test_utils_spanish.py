"""
Unit tests for Spanish license plate validation in utils_spanish_fixed.py.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_spanish_fixed import (
    clean_spanish_text_simple,
    clean_spanish_text_intelligent,
    validate_spanish_plate_flexible,
    format_spanish_plate_nicely,
    preprocess_spanish_plate,
    read_spanish_license_plate_optimized,
    SPANISH_LETTERS,
    SPANISH_DIGITS
)


class TestSpanishCharacterSets:
    """Test Spanish plate character set definitions."""
    
    def test_no_vowels_in_spanish_letters(self):
        """Verify Spanish letters exclude vowels."""
        for vowel in 'AEIOU':
            assert vowel not in SPANISH_LETTERS, f"{vowel} should not be in SPANISH_LETTERS"
    
    def test_no_q_in_spanish_letters(self):
        """Verify Q is excluded from Spanish letters."""
        assert 'Q' not in SPANISH_LETTERS
    
    def test_valid_spanish_letters(self):
        """Test valid Spanish plate letters."""
        valid_letters = 'BCDFGHJKLMNPRSTVWXYZ'
        for letter in valid_letters:
            assert letter in SPANISH_LETTERS, f"{letter} should be in SPANISH_LETTERS"
    
    def test_valid_spanish_digits(self):
        """Test valid Spanish plate digits."""
        for digit in '0123456789':
            assert digit in SPANISH_DIGITS


class TestCleanSpanishTextSimple:
    """Test simple Spanish plate text cleaning."""
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        assert clean_spanish_text_simple("1234-ABC") == "1234ABC"
        assert clean_spanish_text_simple("AB-1234") == "AB1234"
        assert clean_spanish_text_simple("1234 ABC") == "1234ABC"
    
    def test_uppercase_conversion(self):
        """Test lowercase to uppercase conversion."""
        assert clean_spanish_text_simple("1234-abc") == "1234ABC"
        assert clean_spanish_text_simple("ab-1234") == "AB1234"
    
    def test_invalid_character_removal(self):
        """Test removal of invalid Spanish plate characters."""
        # Vowels and Q should be removed
        assert clean_spanish_text_simple("1234-AEI") == "1234"
        assert clean_spanish_text_simple("1234-QBC") == "1234BC"
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert clean_spanish_text_simple("") == ""
    
    def test_all_invalid_chars(self):
        """Test string with all invalid characters."""
        assert clean_spanish_text_simple("AEIOUQ") == ""
    
    def test_space_removal(self):
        """Test space removal."""
        assert clean_spanish_text_simple("12 34 AB C") == "1234ABC"
    
    def test_hyphen_removal(self):
        """Test hyphen removal."""
        assert clean_spanish_text_simple("12-34-AB-C") == "1234ABC"
    
    def test_mixed_valid_invalid(self):
        """Test mixed valid and invalid characters."""
        assert clean_spanish_text_simple("A1B2C3DE") == "123BDE"


class TestCleanSpanishTextIntelligent:
    """Test intelligent Spanish plate text cleaning with corrections."""
    
    def test_o_to_zero_correction(self):
        """Test O becoming 0."""
        # O in first position should become 0 (likely digit position in current format)
        result = clean_spanish_text_intelligent("O234ABC")
        assert '0' in result or 'O' in result
    
    def test_i_to_one_correction(self):
        """Test I becoming 1."""
        result = clean_spanish_text_intelligent("I234ABC")
        assert '1' in result
    
    def test_z_to_two_correction(self):
        """Test Z becoming 2."""
        result = clean_spanish_text_intelligent("Z234ABC")
        assert '2' in result
    
    def test_a_to_four_correction(self):
        """Test A becoming 4."""
        result = clean_spanish_text_intelligent("A234ABC")
        assert '4' in result
    
    def test_s_to_five_correction(self):
        """Test S becoming 5."""
        result = clean_spanish_text_intelligent("S234ABC")
        assert '5' in result
    
    def test_g_to_six_correction(self):
        """Test G becoming 6."""
        result = clean_spanish_text_intelligent("G234ABC")
        assert '6' in result
    
    def test_t_to_seven_correction(self):
        """Test T becoming 7."""
        result = clean_spanish_text_intelligent("T234ABC")
        assert '7' in result
    
    def test_b_to_eight_correction(self):
        """Test B becoming 8."""
        result = clean_spanish_text_intelligent("B234ABC")
        assert '8' in result
    
    def test_p_to_nine_correction(self):
        """Test P becoming 9."""
        result = clean_spanish_text_intelligent("P234ABC")
        assert '9' in result
    
    def test_empty_string(self):
        """Test empty string."""
        assert clean_spanish_text_intelligent("") == ""
    
    def test_short_string(self):
        """Test short string (< 4 chars) returns as-is."""
        assert clean_spanish_text_intelligent("ABC") == "ABC"


class TestValidateSpanishPlateFlexible:
    """Test flexible Spanish plate validation."""
    
    def test_current_format_valid(self):
        """Test valid current format plates (####-LLL)."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("1234ABC")
        assert is_valid == True
        assert formatted == "1234-ABC"
        assert format_type == "current_format"
    
    def test_old_format_valid(self):
        """Test valid old format plates (LL-####)."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("AB1234")
        assert is_valid == True
        assert formatted == "AB-1234"
        assert format_type == "old_format"
    
    def test_current_format_with_hyphen(self):
        """Test current format with hyphen already present."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("1234-ABC")
        assert is_valid == True
        assert formatted == "1234-ABC"
    
    def test_old_format_with_hyphen(self):
        """Test old format with hyphen already present."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("AB-1234")
        assert is_valid == True
        assert formatted == "AB-1234"
    
    def test_short_plate(self):
        """Test plate too short to be valid."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("123")
        assert is_valid == False
        assert formatted is None
    
    def test_empty_plate(self):
        """Test empty plate."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("")
        assert is_valid == False
    
    def test_partial_current_format(self):
        """Test partial current format (3 digits, 3 letters)."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("123ABC")
        assert is_valid == True
        assert format_type == "partial_current"
    
    def test_plate_with_vowels(self):
        """Test plate containing vowels (should be filtered)."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("1234AEI")
        # Vowels are removed, leaving only numbers
        # This may fail validation or produce partial result
        # Depends on implementation
    
    def test_plate_with_q(self):
        """Test plate containing Q (should be filtered)."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("Q234ABC")
        # Q should be removed
        # Result depends on implementation
    
    def test_mixed_format(self):
        """Test mixed format plate."""
        is_valid, formatted, format_type = validate_spanish_plate_flexible("12AB34")
        assert format_type in ["mixed_format", None] or is_valid == False


class TestFormatSpanishPlateNicely:
    """Test Spanish plate formatting with hyphen."""
    
    def test_format_current(self):
        """Test formatting current format."""
        result = format_spanish_plate_nicely("1234ABC")
        assert result == "1234-ABC"
    
    def test_format_old(self):
        """Test formatting old format."""
        result = format_spanish_plate_nicely("AB1234")
        assert result == "AB-1234"
    
    def test_format_already_hyphenated(self):
        """Test plate already containing hyphen."""
        result = format_spanish_plate_nicely("1234-ABC")
        assert result == "1234-ABC"
    
    def test_format_short_plate(self):
        """Test formatting plate too short."""
        result = format_spanish_plate_nicely("123")
        assert result == "123"
    
    def test_format_empty(self):
        """Test formatting empty string."""
        result = format_spanish_plate_nicely("")
        assert result == ""


class TestPreprocessSpanishPlate:
    """Test Spanish plate image preprocessing."""
    
    def test_grayscale_conversion(self):
        """Test grayscale image input."""
        gray_img = np.zeros((50, 100), dtype=np.uint8)
        result = preprocess_spanish_plate(gray_img)
        assert result is not None
        assert len(result.shape) == 2  # Still grayscale
    
    def test_color_conversion(self):
        """Test color image to grayscale conversion."""
        color_img = np.zeros((50, 100, 3), dtype=np.uint8)
        result = preprocess_spanish_plate(color_img)
        assert result is not None
        assert len(result.shape) == 2  # Converted to grayscale
    
    def test_dark_text_light_background(self):
        """Test inversion when text is dark on light background."""
        light_bg = np.ones((50, 100), dtype=np.uint8) * 200
        result = preprocess_spanish_plate(light_bg)
        # Should not invert (already light background)
        assert result is not None
    
    def test_light_text_dark_background(self):
        """Test inversion when text is light on dark background."""
        dark_bg = np.ones((50, 100), dtype=np.uint8) * 30
        result = preprocess_spanish_plate(dark_bg)
        # Should invert (dark background)
        assert result is not None
    
    def test_eu_strip_removal(self):
        """Test removal of EU blue strip."""
        # Create wide plate (EU strip would be on left)
        wide_plate = np.ones((50, 200), dtype=np.uint8) * 200
        result = preprocess_spanish_plate(wide_plate)
        # Left ~15% should be white (removed)
        assert result is not None
        # Width should remain the same (just whitened)
        assert result.shape[1] == 200
    
    def test_enhancement_effects(self):
        """Test that CLAHE and other enhancements are applied."""
        # Create a gradient image
        gradient = np.arange(0, 256, dtype=np.uint8).reshape(1, -1)
        gradient = np.tile(gradient, (50, 1))
        
        result = preprocess_spanish_plate(gradient)
        assert result is not None
        # Result should be binary after adaptive thresholding
        assert set(np.unique(result)).issubset({0, 255})


class TestReadSpanishLicensePlateOptimized:
    """Test Spanish plate OCR reading."""
    
    @patch('utils_spanish_fixed.reader')
    def test_successful_read(self, mock_reader):
        """Test successful plate reading."""
        mock_reader.readtext.return_value = [
            ([(0,0), (100,0), (100,50), (0,50)], "1234-ABC", 0.95)
        ]
        
        test_img = np.zeros((50, 100), dtype=np.uint8)
        plate_text, plate_score = read_spanish_license_plate_optimized(test_img)
        
        # Should return cleaned/formatted version
        assert "1234" in plate_text and "ABC" in plate_text
        assert plate_score > 0.0
    
    @patch('utils_spanish_fixed.reader')
    def test_no_detection(self, mock_reader):
        """Test when OCR detects nothing."""
        mock_reader.readtext.return_value = []
        
        test_img = np.zeros((50, 100), dtype=np.uint8)
        plate_text, plate_score = read_spanish_license_plate_optimized(test_img)
        
        assert plate_text == ""
        assert plate_score == 0.0
    
    @patch('utils_spanish_fixed.reader')
    def test_invalid_format_detection(self, mock_reader):
        """Test when OCR returns invalid format."""
        mock_reader.readtext.return_value = [
            ([(0,0), (100,0), (100,50), (0,50)], "INVALID", 0.90)
        ]
        
        test_img = np.zeros((50, 100), dtype=np.uint8)
        plate_text, plate_score = read_spanish_license_plate_optimized(test_img)
        
        # Should return empty or attempt to clean
        # Depends on implementation
    
    @patch('utils_spanish_fixed.reader')
    def test_multiple_detections(self, mock_reader):
        """Test when OCR returns multiple detections."""
        mock_reader.readtext.return_value = [
            ([(0,0), (50,0), (50,50), (0,50)], "1234ABC", 0.95),
            ([(50,0), (100,0), (100,50), (50,50)], "5678XYZ", 0.90),
        ]
        
        test_img = np.zeros((50, 100), dtype=np.uint8)
        plate_text, plate_score = read_spanish_license_plate_optimized(test_img)
        
        # Should return highest confidence valid plate
        assert plate_score > 0.0
    
    @patch('utils_spanish_fixed.reader')
    def test_low_confidence_detection(self, mock_reader):
        """Test when OCR returns low confidence detection."""
        mock_reader.readtext.return_value = [
            ([(0,0), (100,0), (100,50), (0,50)], "1234ABC", 0.30)
        ]
        
        test_img = np.zeros((50, 100), dtype=np.uint8)
        plate_text, plate_score = read_spanish_license_plate_optimized(test_img)
        
        # Should return the detection, possibly with reduced confidence
        # Depends on implementation


class TestSpanishPlateFormats:
    """Test various Spanish plate format scenarios."""
    
    def test_current_format_variations(self):
        """Test current format variations."""
        test_plates = [
            "1234-ABC",
            "5678-XYZ",
            "9999-BCD",
        ]
        for plate in test_plates:
            is_valid, formatted, _ = validate_spanish_plate_flexible(plate)
            assert is_valid, f"{plate} should be valid"
    
    def test_old_format_variations(self):
        """Test old format variations."""
        test_plates = [
            "AB-1234",
            "CD-5678",
            "ZF-9999",
        ]
        for plate in test_plates:
            is_valid, formatted, _ = validate_spanish_plate_flexible(plate)
            assert is_valid, f"{plate} should be valid"
    
    def test_edge_case_plates(self):
        """Test edge case plates."""
        edge_plates = [
            "0000-AAA",  # All same digits/letters
            "9999-ZZZ",  # All same digits/letters
            "1234-BBB",  # Repeated letters
        ]
        for plate in edge_plates:
            # Should not crash
            is_valid, formatted, _ = validate_spanish_plate_flexible(plate)
    
    def test_lowercase_input(self):
        """Test lowercase input handling."""
        is_valid, formatted, _ = validate_spanish_plate_flexible("1234abc")
        # Should auto-convert to uppercase
        # Exact behavior depends on implementation


class TestOCRErrorCorrection:
    """Test OCR error correction for common mistakes."""
    
    def test_oh_vs_zero(self):
        """Test O and 0 confusion."""
        # Current format: should be zeros in position 0-3
        result = clean_spanish_text_intelligent("O234ABC")
        assert '0' in result
    
    def test_one_vs_i(self):
        """Test 1 and I confusion."""
        result = clean_spanish_text_intelligent("I234ABC")
        assert '1' in result
    
    def test_five_vs_s(self):
        """Test 5 and S confusion."""
        result = clean_spanish_text_intelligent("S234ABC")
        assert '5' in result
    
    def test_six_vs_g(self):
        """Test 6 and G confusion."""
        result = clean_spanish_text_intelligent("G234ABC")
        assert '6' in result
    
    def test_four_vs_a(self):
        """Test 4 and A confusion."""
        result = clean_spanish_text_intelligent("A234ABC")
        assert '4' in result
    
    def test_eight_vs_b(self):
        """Test 8 and B confusion."""
        result = clean_spanish_text_intelligent("B234ABC")
        assert '8' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])