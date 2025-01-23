import random

_ALL_SYMBOLS = ["()", "{}", "[]", "<>"]

def _generate_legal_dyck_string(symbol_options, length, max_nesting_level, current_level=0, must_reach_max_depth=True):
    """
    Helper to generate a valid Dyck string with a given nesting level and length,
    ensuring the maximum nesting level is reached at least once.
    """
    if length <= 0:
        return "", 0

    if max_nesting_level - current_level <= 0:
        # No more nesting allowed, just add pairs
        pairs = random.choices(symbol_options, k=length // 2)
        return "".join([p[0] + p[1] for p in pairs]), 0

    max_depth_from_here = current_level
    result = []
    while len(result) < length:
        remaining_length = length - len(result)

        if must_reach_max_depth and remaining_length <= max_nesting_level * 2:
            # Force reaching max depth before running out of space
            symbol = random.choice(symbol_options)
            result.append(symbol[0])
            inner_string, inner_depth = _generate_legal_dyck_string(symbol_options, remaining_length - 2, max_nesting_level, current_level + 1, must_reach_max_depth)
            result.append(inner_string)
            max_depth_from_here += inner_depth
            if max_depth_from_here >= max_nesting_level: 
                must_reach_max_depth = False
            result.append(symbol[1])
        else:
            if random.random() < 0.5 and current_level < max_nesting_level and remaining_length > 3:
                # Randomly decide to nest further
                symbol = random.choice(symbol_options)
                inner_length = random.randint(1, remaining_length - 3)
                result.append(symbol[0])
                inner_string, inner_depth = _generate_legal_dyck_string(symbol_options, inner_length, max_nesting_level, current_level + 1, must_reach_max_depth)
                result.append(inner_string)
                max_depth_from_here += inner_depth
                if max_depth_from_here >= max_nesting_level: 
                    must_reach_max_depth = False
                result.append(symbol[1])
            else:
                # Add a simple pair
                symbol = random.choice(symbol_options)
                result.append(symbol[0])
                result.append(symbol[1])
    
    # Final sanity check: If we still haven't reached max depth, enforce it
    # if must_reach_max_depth:
    #     symbol = random.choice(symbol_options)
    #     result.append(symbol[0])
    #     result.append(_generate_legal_dyck_string(symbol_options, max_nesting_level * 2 - 2, max_nesting_level, current_level + 1, False))
    #     result.append(symbol[1])

    return "".join(result), max_depth_from_here

def generate_random(num_symbols, length, nesting_level):
    """
    Generates a random string that can either belong to the Dyck language or be invalid.
    
    Args:
        length (int): Target length of the string.
        nesting_level (int): Maximum allowed depth of nesting for valid Dyck strings.

    Returns:
        str: A generated string based on the specified parameters.
    """
    if length <= 0:
        return "", True

    symbol_options = random.sample(_ALL_SYMBOLS, num_symbols)
    
    if random.choice([0, 1]) == 1:
        # Generate a valid Dyck string ensuring the max nesting level is reached
        dyck_string, _ = _generate_legal_dyck_string(symbol_options, length, nesting_level)
        return dyck_string, True
    else:
        # Generate an invalid Dyck string
        dyck_string = _generate_invalid_dyck_string(symbol_options, length, nesting_level)
        return dyck_string, False


def _generate_invalid_dyck_string(symbol_options, length, max_nesting_level):
    """
    Helper to generate an invalid Dyck string by introducing controlled mistakes.
    """
    # Start with a valid Dyck string
    valid_string, _ = _generate_legal_dyck_string(symbol_options, length, max_nesting_level)
    valid_list = list(valid_string)
    
    # Decide on the number of mistakes to introduce
    num_errors = max(1, random.randint(1, length // 4))  # At least one mistake
    
    for _ in range(num_errors):
        try:
            error_type = random.choice(["mismatch", "extra_open", "extra_close", "swap"])
            
            if error_type == "mismatch":
                # Replace one of the brackets with a mismatched one
                idx = random.randint(0, len(valid_list) - 1)
                if valid_list[idx] in [s[0] for s in symbol_options]:  # If it's an opening bracket
                    valid_list[idx] = random.choice([s[0] for s in symbol_options if s[0] != valid_list[idx]])
                elif valid_list[idx] in [s[1] for s in symbol_options]:  # If it's a closing bracket
                    valid_list[idx] = random.choice([s[1] for s in symbol_options if s[1] != valid_list[idx]])
            
            elif error_type == "extra_open":
                # Insert an unmatched opening bracket
                symbol = random.choice([s[0] for s in symbol_options])
                idx = random.randint(0, len(valid_list))
                valid_list.insert(idx, symbol)
            
            elif error_type == "extra_close":
                # Insert an unmatched closing bracket
                symbol = random.choice([s[1] for s in symbol_options])
                idx = random.randint(0, len(valid_list))
                valid_list.insert(idx, symbol)
            
            elif error_type == "swap":
                # Swap two characters to mess up the ordering
                if len(valid_list) > 1:
                    idx1, idx2 = 0, 0
                    while valid_list[idx1] == valid_list[idx2]:
                        idx1, idx2 = random.sample(range(len(valid_list)), 2)
                    valid_list[idx1], valid_list[idx2] = valid_list[idx2], valid_list[idx1]
        except:
            # print(f"Couldn't do the {error_type}, skipping one.")
            pass
    
    return "".join(valid_list)[:length]