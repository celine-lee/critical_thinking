import random

_ALL_OPERATORS = ["and", "or", "not", "xor"]

def generate_random_helper(operator_options, nesting_level):
    if nesting_level == 1:
        operator = random.choice(operator_options)
        rside = random.choice(("True", "False"))
        if operator in {"not"}:
            return f"{operator} {rside}"
        else:
            lside = random.choice(("True", "False"))
            return f"{lside} {operator} {rside}"
    operator = random.choice(operator_options)
    rside = generate_random_helper(operator_options, nesting_level - 1)
    if operator in {"not"}:
        return f"{operator} ({rside})"
    else:
        lside = generate_random_helper(operator_options, nesting_level - 1)
        return f"({lside}) {operator} ({rside})"


def generate_random(num_diff_ops, nesting_level):
    operator_options = random.sample(_ALL_OPERATORS, num_diff_ops)
    expression = generate_random_helper(operator_options, nesting_level)
    answer = eval(expression.replace("xor", "^"))
    return expression, answer