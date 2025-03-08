import os
import re
import random

class Task:
    def __init__(self, name, answer_extraction_regex):
        self.name = name
        self.answer_extraction_regex = answer_extraction_regex
        self.system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

        self.stop_strings = ["[/ANSWER]"]

    def generate_random(self, how_many, dfa_kwargs):
        # TODO if theres only so many random options, limit up to that
        raise NotImplementedError

    def make_prompt(self, kwargs):
        raise NotImplementedError
        
    def extract_answers(self, model_predictions):
        answers = []
        for model_prediction in model_predictions:
            parsed_answer = None
            for parsed_answer in re.finditer(self.answer_extraction_regex, model_prediction):
                pass # only get the last
            if parsed_answer is None:
                answers.append(None)
            else:
                answers.append(parsed_answer.group(1).rstrip(" ."))
        return answers

    def create_subfolder_name(self):
        raise NotImplementedError

class ArrayIdxTask(Task):
    def  __init__(self):
        super(ArrayIdxTask, self).__init__("array_idx_mult", r'pointer ==\s*(\d+)')
        self.foldername = "array_idx_mult/outputs"

        self.query_template = """You are given a length-{full_size} array and must track the index of a 0-indexed pointer to the array. The pointer undergoes several modifications. The pointer wraps around the length of the array on both ends, so when it reaches {full_size} it becomes 0, when it reaches {full_size_plus_1} it becomes 1, when it reaches -1 it becomes {full_size_minus_1}, etc. What is the index of the pointer after all the modifications are complete? Provide the answer in the range [0, {full_size}).

pointer = 0
{sequence}
"""
        self.generation_instruction = "Provide your final answer following this template: [ANSWER]\npointer == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\npointer == "

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_m{dfa_kwargs['m']}_N{dfa_kwargs['N']}")
        return subfolder

    def random_walk(self, k, m, N):
        def normalize_state(idx, full_wrap):
            num_wrap_actions = 0
            while idx < 0: 
                idx += full_wrap
                num_wrap_actions += 1
            while idx >= full_wrap: 
                idx -= full_wrap
                num_wrap_actions += 1
            return idx, num_wrap_actions
        turns = []
        curr_state = 0
        while len(turns) < N:
            transition = m * random.choice(list(range(-k+1, k)))
            if transition < 0:
                turns.append(f"pointer = pointer - {-transition}")
            else:
                turns.append(f"pointer = pointer + {transition}")
            curr_state, wrapped = normalize_state(curr_state + transition, k * m)
        return turns, curr_state

    def make_prompt(self, generator, k, m, turns):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(full_size=k*m, full_size_plus_1=(k*m)+1, full_size_minus_1=(k*m)-1, sequence="\n".join(turns)) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif generator.tokenizer.chat_template:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(full_size=k*m, full_size_plus_1=(k*m)+1, full_size_minus_1=(k*m)-1, sequence="\n".join(turns)) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                breakpoint()
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(full_size=k*m, full_size_plus_1=(k*m)+1, full_size_minus_1=(k*m)-1, sequence="\n".join(turns)) + "\n"
            prompt += self.generation_instruction
        return prompt

    def generate_random(self, generator, kmn):
        k = kmn['k']
        m = kmn['m']
        N = kmn['N']
        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            turns, true_final_location = self.random_walk(k, m, N)
            prompt = self.make_prompt(generator, k, m, turns)
            prompts.append(prompt)
            true_answers.append(true_final_location)
        return prompts, true_answers

class EvenOddTask(ArrayIdxTask):
    def  __init__(self):
        super(EvenOddTask, self).__init__()
        self.name = "even_odd"
        self.answer_extraction_regex = r'pointer_is_even ==\s*(True|False|0|1)'
        self.foldername = "even_odd_mult/outputs"

        self.query_template =  """You are tracking a pointer into a length-{full_size} array. The pointer is zero-indexed. It undergoes several modifications. The pointer wraps around the length of the array on both ends, so when it reaches {full_size} it becomes 0, when it reaches {full_size_plus_1} it becomes 1, when it reaches -1 it becomes {full_size_minus_1}, etc. After all the modifications are complete, is the final pointer index even?

pointer = 0
{sequence}
"""
        self.generation_instruction = "Provide your final answer as True or False following this template: [ANSWER]\npointer_is_even == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\npointer_is_even == "

    def generate_random(self, generator, kmn):
        k = kmn['k']
        m = kmn['m']
        N = kmn['N']
        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            turns, true_final_location = self.random_walk(k, m, N)
            loc_is_even = true_final_location % 2 == 0
            prompt = self.make_prompt(generator, k, m, turns)
            prompts.append(prompt)
            true_answers.append(loc_is_even)
        return prompts, true_answers

class DyckNTask(Task):
    def __init__(self):
        super(DyckNTask, self).__init__("dyck", r'is_balanced\s*==\s*(True|False|0|1)')
        self.foldername = "dyck/outputs"
        self.query_template = """Determine whether the following string belongs to the Dyck language, i.e. is a balanced string of brackets such that every single open bracket has a corresponding closed bracket later in the string.

Input: {dyck_word}
"""
        self.generation_instruction = "Provide your final answer as True or False, following this template: [ANSWER]\nis_balanced == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nis_balanced == "

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_d{dfa_kwargs['d']}_N{dfa_kwargs['N']}")
        return subfolder

    def _generate_legal_dyck_string(self, symbol_options, length, max_nesting_level, current_level=0, must_reach_max_depth=True):
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
                inner_string, inner_depth = self._generate_legal_dyck_string(symbol_options, remaining_length - 2, max_nesting_level, current_level + 1, must_reach_max_depth)
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
                    inner_string, inner_depth = self._generate_legal_dyck_string(symbol_options, inner_length, max_nesting_level, current_level + 1, must_reach_max_depth)
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

    def _generate_invalid_dyck_string(self, symbol_options, length, max_nesting_level):
        """
        Helper to generate an invalid Dyck string by introducing controlled mistakes.
        """
        # Start with a valid Dyck string
        valid_string, _ = self._generate_legal_dyck_string(symbol_options, length, max_nesting_level)
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

    def generate_random_dyck_example(self, nesting_level, length, num_symbols):
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

        _ALL_SYMBOLS = ["()", "{}", "[]", "<>"]
        symbol_options = random.sample(_ALL_SYMBOLS, num_symbols)
        
        if random.choice([0, 1]) == 1:
            # Generate a valid Dyck string ensuring the max nesting level is reached
            dyck_string, _ = self._generate_legal_dyck_string(symbol_options, length, nesting_level)
            return dyck_string, True
        else:
            # Generate an invalid Dyck string
            dyck_string = self._generate_invalid_dyck_string(symbol_options, length, nesting_level)
            return dyck_string, False

    def generate_random(self, generator, kdN):
        nesting_level = kdN['k']
        num_symbols = kdN['d']
        length = kdN['N']

        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            dyck_word, true_answer = self.generate_random_dyck_example(nesting_level, length, num_symbols)

            prompt = self.make_prompt(generator, dyck_word)
            prompts.append(prompt)
            true_answers.append(true_answer)
        return prompts, true_answers

    def make_prompt(self, generator, dyck_word):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(dyck_word=dyck_word) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(dyck_word=dyck_word) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(dyck_word=dyck_word) + "\n"
            prompt += self.generation_instruction
        return prompt


class NestedBoolTask(Task):
    def  __init__(self):
        super(NestedBoolTask, self).__init__("bool", r'truth_value\s*==\s*(True|False|0|1)\s*')
        self.foldername = "nested_boolean_expression/outputs"

        self.query_template = """Evaluate the following boolean expression:

truth_value = {expression}
"""
        self.generation_instruction = "Provide your final answer following this template: [ANSWER]\ntruth_value == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\ntruth_value == "

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_N{dfa_kwargs['N']}")
        return subfolder

    def make_prompt(self, generator, expression):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(expression=expression) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif generator.tokenizer.chat_template:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(expression=expression) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                breakpoint()
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(expression=expression) + "\n"
            prompt += self.generation_instruction
        return prompt
    
    def generate_random_helper(self, operator_options, nesting_level):
        if nesting_level == 1:
            operator = random.choice(operator_options)
            rside = random.choice(("True", "False"))
            if operator in {"not"}:
                return f"{operator} {rside}"
            else:
                lside = random.choice(("True", "False"))
                return f"{lside} {operator} {rside}"
        operator = random.choice(operator_options)
        rside = self.generate_random_helper(operator_options, nesting_level - 1)
        if operator in {"not"}:
            return f"{operator} ({rside})"
        else:
            lside = self.generate_random_helper(operator_options, nesting_level - 1)
            return f"({lside}) {operator} ({rside})"

    def generate_random(self, generator, kN):
        _ALL_OPERATORS = ["and", "or", "not", "xor"]
        num_diff_ops = kN['k']
        nesting_level = kN['N']
        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            operator_options = random.sample(_ALL_OPERATORS, num_diff_ops)
            expression = self.generate_random_helper(operator_options, nesting_level)
            answer = eval(expression.replace("xor", "^"))
            
            prompt = self.make_prompt(generator, expression)
            prompts.append(prompt)
            true_answers.append(answer)
        return prompts, true_answers

class NavigateTask(Task):
    def  __init__(self):
        super(NavigateTask, self).__init__("bool", r'returned_to_start\s*==\s*(True|False|0|1)\s*')
        self.foldername = "navigate/outputs"

        self.query_template = """If you follow these instructions, do you return to the starting point? Always face forward. 

{sequence}
"""
        self.generation_instruction = "Provide your final answer as True or False, following this template: [ANSWER]\nreturned_to_start == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nreturned_to_start == "

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_d{dfa_kwargs['d']}_N{dfa_kwargs['N']}")
        return subfolder

    def make_prompt(self, generator, sequence):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(sequence=sequence) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif generator.tokenizer.chat_template:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(sequence=sequence) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                breakpoint()
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(sequence=sequence) + "\n"
            prompt += self.generation_instruction
        return prompt
    
    def generate_random_helper(self, max_distance_away, target_length, current_length, curr_position, end_at_start):
        """
        Generates a random sequence that either ends at the origin or does not.
        
        Args:
            max_distance_away (int): Size of the world in terms of no. steps away from the origin.
            target_length (int): Target number of turns.
            current_length (int): Current number of steps taken.
            curr_position (List[int]): dimensions (1, 2, or 3)-size tuple describing position before adding this step
            end_at_start (bool): Whether we have to end at the start

        Returns:
            List[str]: List of strings describing the navigation from this step
            bool: ends at start
        """
        directions = [("forward", "backwards"), ("right", "left"), ("up", "down")]
        template = "Take {num_steps} step{multiplier_s} {direction}."

        if current_length >= target_length:
            return [], all(dim == 0 for dim in curr_position)

        if end_at_start and sum(offset != 0 for offset in curr_position) >= target_length - current_length:
            dim_to_move = random.choice([dim for dim, offset in enumerate(curr_position) if offset != 0])
            amount_to_move = 0 - curr_position[dim_to_move]
        elif end_at_start and (target_length - current_length < 2):
            dim_to_move = random.choice(list(range(len(curr_position))))
            amount_to_move = 0
        else:
            dim_to_move = random.choice(list(range(len(curr_position))))
            if curr_position[dim_to_move] >= max_distance_away:
                amount_to_move = random.choice(list(range(-2 * max_distance_away, 0)))
            elif curr_position[dim_to_move] <= -max_distance_away:
                amount_to_move = random.choice(list(range(0, 2 * max_distance_away)))
            else:
                can_move_left = - (max_distance_away + curr_position[dim_to_move])
                can_move_right = max_distance_away - curr_position[dim_to_move]
                amount_to_move = random.choice(list(range(can_move_left, can_move_right)))

        new_position = list(curr_position)
        new_position[dim_to_move] += amount_to_move
        
        if amount_to_move > 0: direction_idx = 0
        else: direction_idx = 1

        rest_of_navigation, actually_ends_at_start = self.generate_random_helper(max_distance_away, target_length, current_length + 1, new_position, end_at_start)
        num_steps = amount_to_move if amount_to_move >=0 else -amount_to_move
        multiplier_s = '' if num_steps == 1 else 's'
        return [template.format(num_steps=num_steps, multiplier_s=multiplier_s, direction=directions[dim_to_move][direction_idx])] + rest_of_navigation, actually_ends_at_start

    def generate_random(self, generator, kdN):
        max_distance_away = kdN['k']
        num_dimensions = kdN['d']
        target_length = kdN['N']
        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            end_at_start = random.choice([False, True])
            sequence, actually_ends_at_start = self.generate_random_helper(max_distance_away, target_length, 0, [0 for _ in range(num_dimensions)], end_at_start)
            prompt = self.make_prompt(generator, sequence)
            prompts.append(prompt)
            true_answers.append(actually_ends_at_start)
        return prompts, true_answers


class CRUXEvalTask(Task):
    def __init__(self):
        super(CRUXEvalTask, self).__init__("cruxeval_straightlined", r'assert answer\s*==\s*(.+)')
        self.foldername = "cruxeval/outputs_straightlined"
        self.query_template = """You are given a snippet of Python code. Complete the assertion with the resulting value in `answer`.

{fn_def}

assert answer == ??
"""
        self.generation_instruction = "Provide your final answer following this template: [ANSWER]\nassert answer == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nassert answer == "

    def create_subfolder_name(self, force_no_cot):
        subfolder = f"{self.foldername}{'_nocot' if force_no_cot else ''}"
        return subfolder

    def generate_random(self, generator, kdN):
        nesting_level = kdN['k']
        num_symbols = kdN['d']
        length = kdN['N']

        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            dyck_word, true_answer = self.generate_random_dyck_example(nesting_level, length, num_symbols)

            prompt = self.make_prompt(generator, dyck_word)
            prompts.append(prompt)
            true_answers.append(true_answer)
        return prompts, true_answers

    def make_prompt(self, generator, fn_def, input_str):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(fn_def=fn_def) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(fn_def=fn_def) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(fn_def=fn_def) + "\n"
            prompt += self.generation_instruction
        return prompt


class ArithmeticTask(NestedBoolTask):
    # https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/multistep_arithmetic_two.txt
    # each parentheses will always be 3 ops, 4 vals
    def  __init__(self):
        super(ArithmeticTask, self).__init__()
        self.name = "arithmetic"
        self.answer_extraction_regex = r'answer\s*==\s*(\d+)\s*'
        self.foldername = "arithmetic/outputs"

        self.query_template = """Solve the following multi-step arithmetic problem:

answer = {expression}
"""
        self.generation_instruction = "Provide your final answer following this template: [ANSWER]\nanswer == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nanswer == "

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_m{dfa_kwargs['m']}_N{dfa_kwargs['N']}")
        return subfolder
    
    def generate_random_helper(self, operator_options, number_range, num_steps):
        if num_steps == 1:
            operators = random.choices(operator_options, k=3)
            numbers = random.choices(list(range(number_range)), k=4)
            return f"{numbers[0]} {operators[0]} {numbers[1]} {operators[1]} {numbers[2]} {operators[2]} {numbers[3]}"
        operator = random.choice(operator_options)
        rside = self.generate_random_helper(operator_options, number_range, num_steps - 1)
        lside = self.generate_random_helper(operator_options, number_range, num_steps - 1)
        return f"({lside}) {operator} ({rside})"

    def generate_random(self, generator, kmN):
        _ALL_OPERATORS = ["+", "*", "-"]
        num_diff_ops = kmN['k']
        number_range = kmN['m']
        num_steps = kmN['N']
        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            operator_options = random.sample(_ALL_OPERATORS, num_diff_ops)
            expression = self.generate_random_helper(operator_options, number_range, num_steps)
            answer = eval(expression)
            
            prompt = self.make_prompt(generator, expression)
            prompts.append(prompt)
            true_answers.append(answer)
        return prompts, true_answers

# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/tracking_shuffled_objects_three_objects.json
class ShuffledObjectsTask(Task):
    def  __init__(self):
        super(ShuffledObjectsTask, self).__init__("shuffled_objects", r'Answer\s*:\s*(.+)')
        self.foldername = "shuffled_objects/outputs"

        self.query_template = """{intro} {sequence}
{final_question}"""
        self.generation_instruction = "Provide your final answer following this template: [ANSWER]\nAnswer: YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nAnswer: "

        self.all_names = ["Alice", "Bob", "Claire", "Ophelia", "Lola", "Izzi", "Helga"]

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_N{dfa_kwargs['N']}")
        return subfolder

    def generate_random_example(self, num_objects, num_swaps, random_num):
        match random_num:
            case 0: #books
                intro = "{names} are friends and avid readers who occasionally trade books. At the start of the semester, they each buy one new book: {initial_assignments}.\nAs the semester proceeds, they start trading around the new books. "
                initial_assignments_template = "{name} gets {object}"
                swap_template = "{order}, {name_1} and {name_2} swap books."
                final_question = "At the end of the semester, which book does {name} have?"
                selected_objects = random.sample(["Moby Dick", "The Great Gatsby", "Frankenstein", "Ulysses", "The Pearl", "The Fellowship of the Ring", "Catch-22"], k=num_objects)
            case 1: #dancers
                intro = "{names} are dancers at a square dance. At the start of a song, they each have a partner: {initial_assignments}.\nThroughout the song, the dancers often trade partners. "
                initial_assignments_template = "{name} is dancing with {object}"
                swap_template = "{order}, {name_1} and {name_2} switch partners."
                final_question = "At the end of the dance, who is {name} dancing with?"
                partners = ["Sam", "Patrick", "Jamie", "Melissa", "Rodrigo", "Stacey", "George"]
                selected_objects = random.sample(partners, k=num_objects)
            case 2: #game
                intro = "{names} are playing a game. At the start of the game, they are each holding a ball: {initial_assignments}.\nAs the game progresses, pairs of players trade balls. "
                initial_assignments_template = "{name} has a {object} ball"
                swap_template = "{order}, {name_1} and {name_2} swap balls."
                final_question = "At the end of the game, what color ball is {name} holding?"
                colors = ["red", "white", "blue", "black", "orange", "green", "yellow", "purple"]
                selected_objects = random.sample(colors, k=num_objects)
            case 3: #present
                intro = "{names} are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: {initial_assignments}.\nAs the event progresses, pairs of people swap gifts. "
                initial_assignments_template = "{name} has a {object} present"
                swap_template = "{order}, {name_1} and {name_2} swap their gifts."
                final_question = "At the end of the event, what color gift does {name} have?"
                colors = ["red", "white", "blue", "black", "orange", "green", "yellow", "purple"]
                selected_objects = random.sample(colors, k=num_objects)
            case 4: #soccer
                intro = "{names} are on the same team in a soccer match. At the start of the match, they are each assigned to a position: {initial_assignments}.\nAs the game progresses, pairs of players occasionally swap positions. "
                initial_assignments_template = "{name} is playing {object}"
                swap_template = "{order}, {name_1} and {name_2} trade positions."
                final_question = "At the end of the match, what position is {name} playing?"
                positions = ["cheerleader", "right winger", "left winger", "goalkeeper", "center midfielder", "benchwarmer", "striker", "fullback"]
                selected_objects = random.sample(positions, k=num_objects)

        selected_names = random.sample(self.all_names, k=num_objects)
        final_name_str = random.choice(selected_names)

        assignments = {name: obj for (name, obj) in zip(selected_names, selected_objects)}
        initial_assignments = [initial_assignments_template.format(name=name, object=obj) for (name, obj) in assignments.items()]
        swaps = []
        while len(swaps) < num_swaps:
            swappers = random.sample(selected_names, k=2)
            temp = assignments[swappers[0]]
            assignments[swappers[0]] = assignments[swappers[1]]
            assignments[swappers[1]] = temp
            if len(swaps) == 0:
                swaps.append(swap_template.format(order="First", name_1=swappers[0], name_2=swappers[1]))
            elif len(swaps) == num_swaps - 1:
                swaps.append(swap_template.format(order="Finally", name_1=swappers[0], name_2=swappers[1]))
            else:
                swaps.append(swap_template.format(order="Then", name_1=swappers[0], name_2=swappers[1]))
        true_answer_str = assignments[final_name_str]
        intro_str = intro.format(
            names=", ".join(selected_names[:-1]) + f", and {selected_names[-1]}", 
            initial_assignments=", ".join(initial_assignments[:-1]) + f", and {initial_assignments[-1]}"
        )
        sequence_str = " ".join(swaps)
        final_question_str = final_question.format(name=final_name_str)
        return intro_str, sequence_str, final_question_str, true_answer_str

    def generate_random(self, generator, kN):
        num_objects = kN['k']
        num_swaps = kN['N']

        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            domain_idx = random.choice(range(5))
            intro_str, sequence_str, final_question_str, true_answer = self.generate_random_example(num_objects, num_swaps, domain_idx)

            prompt = self.make_prompt(generator, intro_str, sequence_str, final_question_str)
            prompts.append(prompt)
            true_answers.append(true_answer)
        return prompts, true_answers

    def make_prompt(self, generator, intro_str, sequence_str, final_question_str):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(intro=intro_str, sequence=sequence_str, final_question=final_question_str) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(intro=intro_str, sequence=sequence_str, final_question=final_question_str) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(intro=intro_str, sequence=sequence_str, final_question=final_question_str) + "\n"
            prompt += self.generation_instruction
        return prompt

# https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/refs/heads/main/bbh/web_of_lies.json
class WebOfLiesTask(Task): # i dont like this one bc it doesnt have k and N.
    def  __init__(self):
        super(WebOfLiesTask, self).__init__("web_of_lies", r'Answer\s*:\s*(.+)')
        self.foldername = "web_of_lies/outputs"

        self.query_template = """Question: {sequence} Does {final_name} tell the truth?"""
        self.generation_instruction = "Provide your final answer as Yes or No, following this template: [ANSWER]\nAnswer: YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nAnswer: "

        self.all_names = ["Sherrie", "Vernell", "Elanor", "Ka", "Delbert", "Jamey", "William", "Sima", "Shaunda", "Tamika", "Teressa", "Lorine", "Conception", "Millicent", "Fletcher", "Alejandro", "Shenna", "Alexa"]
        self.starter = ["{name} lies.", "{name} tells the truth."]
        self.mid = ["{name_1} says {name_2} lies.", "{name_1} says {name_2} tells the truth."]

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_N{dfa_kwargs['N']}")
        return subfolder

    def generate_random_example(self, num_people):
        selected_names = random.sample(self.all_names, k=num_people+1)

        first_lies_or_truths = random.choice([0, 1])
        tells = [self.starter[first_lies_or_truths].format(selected_names[0])]
        tells_truth = first_lies_or_truths
        while len(tells) < num_people:
            says_prev_tells_truth = random.choice([0,1])
            tells_truth = (tells_truth and says_prev_tells_truth) or (not tells_truth and not says_prev_tells_truth)
            name_1 = selected_names[len(tells)]
            name_2 = selected_names[len(tells)-1]
            tells.append(self.mid[lies_or_truths].format(name_1=name_1, name_2=name_2))
            swaps.append(swap_template.format(order="Then", name_1=swappers[0], name_2=swappers[1]))
        sequence_str = " ".join(tells)
        return sequence_str, selected_names[-1], "Yes" if tells_truth else "Noe"

    def generate_random(self, generator, kN):
        num_people = kN['k']

        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            sequence_str, final_name, true_answer = self.generate_random_example(num_people)

            prompt = self.make_prompt(generator, sequence_str, final_name)
            prompts.append(prompt)
            true_answers.append(true_answer)
        return prompts, true_answers

    def make_prompt(self, generator, sequence_str, final_name):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(sequence=sequence_str, final_name=final_name) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(sequence=sequence_str, final_name=final_name) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(sequence=sequence_str, final_name=final_name) + "\n"
            prompt += self.generation_instruction
        return prompt
