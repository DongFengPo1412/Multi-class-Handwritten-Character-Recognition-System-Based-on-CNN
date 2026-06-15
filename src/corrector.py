import os
import numpy as np
import torch
import sys

# Dictionary of characters in EMNIST ByClass (62 classes)
label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
char_to_idx = {char: i for i, char in enumerate(label_map)}

# Confusing pairs (all lowercase for case-insensitive comparisons)
CONFUSING_PAIRS = [
    ('l', 'i'), ('l', '1'), ('i', '1'),
    ('o', '0'), ('o', 'o'),  # symmetric check
    ('z', '2'),
    ('s', '5'),
    ('g', '9'), ('q', '9'), ('g', 'q'),
    ('u', 'v'), ('v', 'w'), ('u', 'w'),
    ('c', 'o'), ('c', 'e')
]

CONFUSING_SET = set()
for a, b in CONFUSING_PAIRS:
    CONFUSING_SET.add((a.lower(), b.lower()))
    CONFUSING_SET.add((b.lower(), a.lower()))

class HandwrittenCorrector:
    def __init__(self, dict_path=None):
        self.dictionary = set()
        self.words_by_len = {}
        
        if dict_path is None:
            # Default path in project
            dict_path = os.path.join(os.path.dirname(__file__), "words.txt")
            
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word and len(word) >= 2 and word.isalpha():
                            self.dictionary.add(word)
                            l = len(word)
                            if l not in self.words_by_len:
                                self.words_by_len[l] = []
                            self.words_by_len[l].append(word)
                print(f"Corrector: Loaded dictionary with {len(self.dictionary)} words.")
            except Exception as e:
                print(f"Corrector Warning: Failed to load dictionary: {e}")
        else:
            print(f"Corrector Warning: Dictionary file not found at {dict_path}")

    def get_context(self, raw_preds):
        """
        Determine context: 'alpha' (letters), 'numeric' (digits), or 'neutral'.
        """
        # Exclude easily confused characters to avoid feedback loop
        clear_digits = set("34678")
        clear_letters = set("abcdefghjkmnpqrtuvwxyABCDEFGHJKMNPQRSTUVWXY")
        
        digit_count = 0
        letter_count = 0
        for char in raw_preds:
            if char in clear_digits:
                digit_count += 1
            elif char in clear_letters:
                letter_count += 1
                
        if letter_count > digit_count:
            return "alpha"
        elif digit_count > letter_count:
            return "numeric"
        return "neutral"

    def apply_geometry_corrections(self, char_probs, aspect_ratios, relative_heights, context):
        """
        Apply aspect ratio and relative height constraints directly to modify the probability distribution.
        This modifies the probabilities *before* decoding to align vision with geometry.
        """
        n = len(char_probs)
        adjusted_probs = []
        
        for i in range(n):
            prob = char_probs[i].clone()
            ar = aspect_ratios[i]
            rh = relative_heights[i]
            
            # 1. Aspect Ratio for 0 vs O/o
            # 0 is typically narrow, O/o is wide.
            # If w/h is very small (< 0.52), it's highly likely to be 0 (zero) or 1.
            # We penalize 'O' and 'o' if the character is very narrow.
            if ar < 0.52:
                prob[char_to_idx['O']] *= 0.05
                prob[char_to_idx['o']] *= 0.05
                # Give a small boost to '0' and '1'
                prob[char_to_idx['0']] *= 1.5
                prob[char_to_idx['1']] *= 1.5
            else:
                # If wide, penalize '0' and '1'
                prob[char_to_idx['0']] *= 0.05
                prob[char_to_idx['1']] *= 0.05
                
            # 2. Relative Height Casing Correction
            # Case-ambiguous characters: C/c, O/o, S/s, U/u, V/v, W/w, X/x, Z/z
            case_ambiguous = [
                ('C', 'c'), ('O', 'o'), ('S', 's'), ('U', 'u'),
                ('V', 'v'), ('W', 'w'), ('X', 'x'), ('Z', 'z')
            ]
            
            # If relative height is low (< 0.78), it's likely lowercase.
            # If relative height is high (>= 0.78), it's likely uppercase.
            is_lowercase_height = rh < 0.78
            
            for upper_c, lower_c in case_ambiguous:
                u_idx = char_to_idx[upper_c]
                l_idx = char_to_idx[lower_c]
                if is_lowercase_height:
                    # Penalize uppercase, boost lowercase
                    prob[u_idx] *= 0.05
                    prob[l_idx] *= 1.8
                else:
                    # Penalize lowercase, boost uppercase
                    prob[l_idx] *= 0.05
                    prob[u_idx] *= 1.8
                    
            # 3. Contextual biasing
            if context == "alpha":
                # Boost letters, penalize digits
                for digit in "0123456789":
                    prob[char_to_idx[digit]] *= 0.1
            elif context == "numeric":
                # Boost digits, penalize letters
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                    prob[char_to_idx[letter]] *= 0.05
                    
            # Normalize back to probability distribution
            prob = prob / (prob.sum() + 1e-9)
            adjusted_probs.append(prob)
            
        return adjusted_probs

    def joint_probability_decode(self, adjusted_probs):
        """
        Perform Joint Probability Lexicon Decoding.
        Finds the word in the dictionary of length N that maximizes:
        Score(W) = sum( log( P(c_i | x_i) ) )
        We aggregate case probabilities: P_case_insensitive(c) = P(c.lower()) + P(c.upper())
        """
        N = len(adjusted_probs)
        if N < 2 or len(self.dictionary) == 0:
            # If too short or dictionary is empty, just do argmax for each character
            decoded = []
            for prob in adjusted_probs:
                idx = torch.argmax(prob).item()
                decoded.append(label_map[idx])
            return "".join(decoded), 1.0
            
        # Get candidate words of length N, or length N-1 / N+1
        candidates = self.words_by_len.get(N, [])
        
        if not candidates:
            # If no candidates of exact length, fall back to argmax
            decoded = []
            for prob in adjusted_probs:
                idx = torch.argmax(prob).item()
                decoded.append(label_map[idx])
            return "".join(decoded), 0.5
            
        best_word = None
        max_log_prob = -9999.0
        
        for word in candidates:
            log_prob_sum = 0.0
            for i, char in enumerate(word):
                prob_char = adjusted_probs[i]
                
                # Case-insensitive probability aggregation
                p_lower = prob_char[char_to_idx.get(char.lower(), 0)].item()
                p_upper = prob_char[char_to_idx.get(char.upper(), 0)].item()
                p_combined = p_lower + p_upper
                
                log_prob_sum += np.log(p_combined + 1e-12)
                
            if log_prob_sum > max_log_prob:
                max_log_prob = log_prob_sum
                best_word = word
                
        # Confidence score estimate based on average probability per character
        avg_prob = np.exp(max_log_prob / N)
        
        # If the best word joint probability is very low, it might be a name or gibberish.
        # If avg_prob < 0.1, we might want to fall back to argmax, but normally we trust the dictionary.
        if avg_prob < 0.02:
            decoded = []
            for prob in adjusted_probs:
                idx = torch.argmax(prob).item()
                decoded.append(label_map[idx])
            return "".join(decoded), avg_prob
            
        # Capitalization reconstruction based on original adjusted probabilities
        final_word = []
        for i, char in enumerate(best_word):
            prob_char = adjusted_probs[i]
            p_lower = prob_char[char_to_idx.get(char.lower(), 0)].item()
            p_upper = prob_char[char_to_idx.get(char.upper(), 0)].item()
            if p_upper > p_lower:
                final_word.append(char.upper())
            else:
                final_word.append(char.lower())
                
        return "".join(final_word), avg_prob

    def decode_sequence(self, raw_probs, aspect_ratios, relative_heights):
        """
        Full decoding pipeline:
        1. Argmax for raw string.
        2. Context classification.
        3. Geometric probability adjustment.
        4. Joint probability decoding for words, or argmax for numbers.
        """
        # Convert to pytorch tensor if they are numpy arrays
        probs = [torch.tensor(p) if not isinstance(p, torch.Tensor) else p for p in raw_probs]
        
        # 1. Get raw string
        raw_chars = [label_map[torch.argmax(p).item()] for p in probs]
        raw_str = "".join(raw_chars)
        
        # 2. Get context
        context = self.get_context(raw_chars)
        
        # 3. Apply geometry & context adjustments
        adjusted_probs = self.apply_geometry_corrections(probs, aspect_ratios, relative_heights, context)
        
        # Calculate minimum character confidence in raw prediction
        raw_confs = [torch.max(p).item() for p in probs]
        min_conf = min(raw_confs) if raw_confs else 0.0

        # 4. Decode
        if min_conf > 0.95:
            # If CNN is extremely confident, bypass dictionary to preserve custom names/acronyms (e.g. CNN, Liu)
            decoded_chars = []
            for p in adjusted_probs:
                idx = torch.argmax(p).item()
                decoded_chars.append(label_map[idx])
            decoded_str = "".join(decoded_chars)
        elif context == "alpha" and len(self.dictionary) > 0:
            decoded_str, conf = self.joint_probability_decode(adjusted_probs)
        else:
            # Numeric or neutral: do argmax on adjusted probabilities
            decoded_chars = []
            for p in adjusted_probs:
                idx = torch.argmax(p).item()
                decoded_chars.append(label_map[idx])
            decoded_str = "".join(decoded_chars)
            conf = 1.0
            
        return raw_str, decoded_str, context
