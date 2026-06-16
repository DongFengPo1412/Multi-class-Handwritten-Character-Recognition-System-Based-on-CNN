import sys
import os
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.corrector import HandwrittenCorrector, label_map, char_to_idx

def test_patterns():
    corrector = HandwrittenCorrector()
    
    # Helper to build dummy probability distribution where target character has 1.0 probability
    def make_prob_distribution(char_probs_dict):
        # char_probs_dict maps character -> probability weight
        prob = torch.zeros(62)
        for char, weight in char_probs_dict.items():
            prob[char_to_idx[char]] = weight
        # Normalize
        s = prob.sum()
        if s > 0:
            prob /= s
        else:
            prob[char_to_idx['0']] = 1.0
        return prob

    # Test cases: (input_chars, aspect_ratios, relative_heights, expected_raw, expected_decoded, expected_pattern)
    cases = [
        # 1. Chinese License Plate: "粤B1O850" (where 粤 is recognized as '8', 'O' as 'O', '0' as '0')
        {
            "chars": [
                {'8': 0.9},             # '粤' misrecognized as '8'
                {'B': 0.8, '8': 0.2},   # 'B'
                {'1': 0.9},             # '1'
                {'O': 0.8, '0': 0.2},   # 'O' (confusing letter)
                {'8': 0.9},             # '8'
                {'5': 0.9},             # '5'
                {'0': 0.9}              # '0'
            ],
            "ars": [0.6, 0.6, 0.3, 0.6, 0.6, 0.6, 0.6],
            "rhs": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "expected_pattern": "license_plate",
            "expected_decoded": "8B10850" # First char 8 (province) preserved, B city, O corrected to 0
        },
        # 2. License Plate without Chinese: "B1O850"
        {
            "chars": [
                {'B': 0.8, '8': 0.2},
                {'1': 0.9},
                {'O': 0.8, '0': 0.2},
                {'8': 0.9},
                {'5': 0.9},
                {'0': 0.9}
            ],
            "ars": [0.6, 0.3, 0.6, 0.6, 0.6, 0.6],
            "rhs": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "expected_pattern": "license_plate_no_cn",
            "expected_decoded": "B10850"
        },
        # 3. ID Card: 18 chars, "11010119900307234x" (ending with lowercase x)
        {
            "chars": [
                {'1': 0.9}, {'1': 0.9}, {'0': 0.9}, {'1': 0.9}, {'0': 0.9}, {'1': 0.9},
                {'1': 0.9}, {'9': 0.9}, {'9': 0.9}, {'0': 0.9}, {'0': 0.9}, {'3': 0.9},
                {'0': 0.9}, {'7': 0.9}, {'2': 0.9}, {'3': 0.9}, {'4': 0.9},
                {'x': 0.7, 'X': 0.1, '4': 0.2}
            ],
            "ars": [0.3]*18,
            "rhs": [1.0]*18,
            "expected_pattern": "id_card",
            "expected_decoded": "11010119900307234X" # x corrected to uppercase X
        },
        # 4. Mobile Phone: 11 chars, "lB912345678" -> "18912345678"
        {
            "chars": [
                {'l': 0.6, '1': 0.4},   # starts with 'l', should be '1'
                {'B': 0.7, '8': 0.3},   # 'B' instead of '8'
                {'9': 0.9}, {'1': 0.9}, {'2': 0.9}, {'3': 0.9}, {'4': 0.9}, {'5': 0.9}, {'6': 0.9}, {'7': 0.9}, {'8': 0.9}
            ],
            "ars": [0.3]*11,
            "rhs": [1.0]*11,
            "expected_pattern": "phone_number",
            "expected_decoded": "18912345678"
        },
        # 5. Meaningless Mixed Alphanumeric: "A7zK29o"
        {
            "chars": [
                {'A': 0.9}, {'7': 0.9}, {'z': 0.9}, {'K': 0.9}, {'2': 0.9}, {'9': 0.9}, {'o': 0.9}
            ],
            "ars": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            "rhs": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "expected_pattern": "neutral",
            "expected_decoded": "A7zK29o"
        },
        # 6. Alphabetical Word with digit confusions: "he11o" -> "hello"
        {
            "chars": [
                {'h': 0.9}, {'e': 0.9}, {'1': 0.7, 'l': 0.3}, {'1': 0.7, 'l': 0.3}, {'o': 0.9}
            ],
            "ars": [0.6, 0.6, 0.3, 0.3, 0.6],
            "rhs": [1.0, 0.7, 1.0, 1.0, 0.7],
            "expected_pattern": "alpha",
            "expected_decoded": "hello"
        },
        # 7. Alphabetical Word with stronger digit confusions: "he110" -> "hello"
        {
            "chars": [
                {'h': 0.9}, {'e': 0.9}, {'1': 0.95}, {'1': 0.95}, {'0': 0.95}
            ],
            "ars": [0.6, 0.6, 0.3, 0.3, 0.6],
            "rhs": [1.0, 0.7, 1.0, 1.0, 0.7],
            "expected_pattern": "alpha",
            "expected_decoded": "hello"
        }
    ]

    all_passed = True
    for idx, case in enumerate(cases, 1):
        probs = [make_prob_distribution(char_dict) for char_dict in case["chars"]]
        raw_str, decoded_str, pattern = corrector.decode_sequence(probs, case["ars"], case["rhs"])
        
        expected_pattern = case["expected_pattern"]
        expected_decoded = case["expected_decoded"]
        
        pattern_ok = (pattern == expected_pattern)
        decoded_ok = (decoded_str == expected_decoded)
        
        print(f"Case {idx}:")
        print(f"  Raw string from CNN  : {raw_str}")
        print(f"  Detected pattern     : {pattern} (Expected: {expected_pattern}) -> {'PASS' if pattern_ok else 'FAIL'}")
        print(f"  Decoded output       : {decoded_str} (Expected: {expected_decoded}) -> {'PASS' if decoded_ok else 'FAIL'}")
        
        if not (pattern_ok and decoded_ok):
            all_passed = False
            
    if all_passed:
        print("\n[+] SUCCESS: All corrector pattern tests passed successfully!")
    else:
        print("\n[-] FAILURE: Some corrector tests failed.")
        sys.exit(1)

if __name__ == '__main__':
    test_patterns()
