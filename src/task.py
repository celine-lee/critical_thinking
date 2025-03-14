import os
import re
import random
import glob
import json

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

{straightlined_code}

assert answer == ??
"""
        self.generation_instruction = "Provide your final answer following this template: [ANSWER]\nassert answer == YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nassert answer == "

        self.all_examples = {}
        self.tracker = {}

    def load_remaining_inputs(self, modelname):

        self.all_examples = {}
        self.tracker = {}
        
        already_processed = {}
        for kN_folder in glob.glob("cruxeval/outputs_straightlined/k*"):
            parsed_experimentname = re.search(r"k(\d+)_N(\d+)", kN_folder)
            if parsed_experimentname is None:
                continue
            k = int(parsed_experimentname.group(1))
            N = int(parsed_experimentname.group(2))
            filename = os.path.join(kN_folder, f"{modelname}_T0.0.json")
            if os.path.exists(filename):
                already_processed[(k, N)] = {ex["id"] for ex in json.load(open(filename))}

        # Then load in the examples that haven't been processed yet; sort accordingly
        for kN_file in glob.glob("cruxeval/synth_cruxeval_straightlined/k*.json"):
            parsed_experimentname = re.search(r"k(\d+)_N(\d+)", kN_file)
            if parsed_experimentname is None:
                continue
            k = int(parsed_experimentname.group(1))
            N = int(parsed_experimentname.group(2))
            self.all_examples[(k,N)] = [ex for ex in json.load(open(kN_file)) if ex["id"] not in already_processed[(k, N)]]
            self.tracker[(k, N)] = 0

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_N{dfa_kwargs['N']}")
        return subfolder

    def make_prompt(self, generator, straightlined_code):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(straightlined_code=straightlined_code) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(straightlined_code=straightlined_code) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(straightlined_code=straightlined_code) + "\n"
            prompt += self.generation_instruction
        return prompt

    def get_example(self, k, N):
        k = int(k)
        N = int(N)
        if self.tracker[(k, N)] >= len(self.all_examples[(k, N)]): return None
        next_ex = self.all_examples[(k, N)][self.tracker[(k, N)]]
        self.tracker[(k, N)] += 1
        return next_ex

    def generate_random(self, generator, kN):
        ast_size = kN["k"]
        trace_len = kN["N"]
        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            ex = self.get_example(ast_size, trace_len)
            prompt = self.make_prompt(generator, ex["straightlined_code"])
            prompts.append(prompt)
            true_answers.append(ex["output"])
        return prompts, true_answers


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
class WebOfLiesTask(Task): # this one, k and N are the same....
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
        selected_names = random.sample(self.all_names, k=num_people)

        first_lies_or_truths = random.choice([0, 1])
        tells = [self.starter[first_lies_or_truths].format(name=selected_names[0])]
        tells_truth = first_lies_or_truths
        while len(tells) < num_people:
            says_prev_tells_truth = random.choice([0,1])
            tells_truth = (tells_truth and says_prev_tells_truth) or (not tells_truth and not says_prev_tells_truth)
            name_1 = selected_names[len(tells)]
            name_2 = selected_names[len(tells)-1]
            tells.append(self.mid[says_prev_tells_truth].format(name_1=name_1, name_2=name_2))
        sequence_str = " ".join(tells)
        return sequence_str, selected_names[-1], "Yes" if tells_truth else "No"

    def generate_random(self, generator, kN):
        # this one has no k... or they're the same. because it's a chain
        num_people = kN['N']

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

from num2words import num2words
# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/logical_deduction_five_objects.txt
class LogicalDeductionTask(Task):
    def  __init__(self):
        super(LogicalDeductionTask, self).__init__("logical_deduction", r'Answer\s*:\s*(.+)')
        self.foldername = "logical_deduction/outputs"

        self.query_template = """The following is a logical deduction task which requires deducing the order of a sequence of objects.

The following sentences each describe a set of {num_objects} objects arranged in a fixed order. The statements are logically consistent within each paragraph. {intro} {sequence} {final_question}"""
        self.generation_instruction = "Provide your final answer following this template: [ANSWER]\nAnswer: YOUR ANSWER\n[/ANSWER]"
        self.reprompt_string = "[ANSWER]\nAnswer: "

    def create_subfolder_name(self, dfa_kwargs, force_no_cot):
        subfolder = os.path.join(f"{self.foldername}{'_nocot' if force_no_cot else ''}", f"k{dfa_kwargs['k']}_N{dfa_kwargs['N']}")
        return subfolder

    def generate_random_example(self, num_objects, num_steps_to_answer, random_num):
        match random_num:
            case 0: #books
                intro_template = "On a shelf, there are {num_objects} books: {objects}."
                relative_order_template = "The {object_1} is {num_spots} to the {direction} of the {object_2}."
                directions = ["left", "right"]
                positions_from_start = ["leftmost", "second from the left", "third from the left", "fourth from the left", "fifth from the left", "sixth from the left", "seventh from the left"]
                positions_from_end = ["sixth from the right", "fifth from the right", "fourth from the right", "third from the right", "second from the right", "rightmost"]
                absolute_order_template = "The {object} is {position}."
                final_question_template = "Which color book is {query_position}?"
                colors = ["red book", "white book", "blue book", "black book", "orange book", "green book", "yellow book", "purple book", "brown book", "indigo book", "pink book", "gray book"]
                selected_objects = random.sample(colors, k=num_objects)
                object_to_answer_formatter = lambda book_name: book_name.split()[0]
            case 1: #car show
                intro_template = "In an antique car show, there are {num_objects} vehicles: {objects}."
                relative_order_template = "The {object_1} is {num_spots} spots {direction} than the {object_2}."
                directions = ["older", "newer"]
                positions_from_start = ["oldest", "second-oldest", "third-oldest", "fourth-oldest", "fifth-oldest", "sixth-oldest", "seventh-oldest"]
                positions_from_end = ["sixth-newest", "fifth-newest", "fourth-newest", "third-newest", "second-newest", "newest"]
                absolute_order_template = "The {object} is {position}."
                final_question_template = "Which vehicle is {query_position}?"
                vehicles = ["hatchback", "bus", "convertible", "tractor", "minivan", "sedan", "golf cart", "scooter", "motorcycle", "limousine", "prius", "truck"]
                selected_objects = random.sample(vehicles, k=num_objects)
                object_to_answer_formatter = lambda vehicle_name: vehicle_name
            case 2: #branch
                intro_template = "On a branch, there are {num_objects} birds: {objects}."
                relative_order_template = "The {object_1} is {num_spots} to the {direction} of the {object_2}."
                directions = ["left", "right"]
                positions_from_start = ["leftmost", "second from the left", "third from the left", "fourth from the left", "fifth from the left", "sixth from the left", "seventh from the left"]
                positions_from_end = ["sixth from the right", "fifth from the right", "fourth from the right", "third from the right", "second from the right", "rightmost"]
                absolute_order_template = "The {object} is {position}."
                final_question_template = "Which bird is {query_position}?"
                birds = ["quail", "owl", "raven", "falcon", "robin", "tit", "pigeon", "eagle", "bluejay", "dove", "chicken", "parrot"]
                selected_objects = random.sample(birds, k=num_objects)
                object_to_answer_formatter = lambda bird_name: bird_name
            case 3: #golf
                intro_template = "In a golf tournament, there were {num_objects} golfers: {objects}."
                relative_order_template = "{object_1} finished {num_spots} spots {direction} {object_2}."
                directions = ["above", "below"]
                positions_from_start = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
                positions_from_end = ["sixth-last", "fifth-last", "fourth-last", "third-last", "second-last", "last"]
                absolute_order_template = "{object} finished {position}."
                final_question_template = "Who finished {query_position}?"
                people = ["Rob", "Ada", "Dan", "Joe", "Mel", "Jack", "Sue", "Pat", "Amy", "Lee", "Jon", "Lev"]
                selected_objects = random.sample(people, k=num_objects)
                object_to_answer_formatter = lambda person_name: person_name
            case 4: #fruit
                intro_template = "A fruit stand sells {num_objects} fruits: {objects}."
                relative_order_template = "{object_1} are {num_spots} dollars {direction} than {object_2}."
                directions = ["cheaper", "more expensive"]
                positions_from_start = ["cheapest", "second-cheapest", "third-cheapest", "fourth-cheapest", "fifth-cheapest", "sixth-cheapest", "seventh-cheapest"]
                positions_from_end = ["sixth-most expensive", "fifth-most expensive", "fourth-most expensive", "third-most expensive", "second-most expensive", "most expensive"]
                absolute_order_template = "The {object} are {position}."
                final_question_template = "Which fruits are {query_position}?"
                fruits = ["kiwis", "pears", "peaches", "loquats", "apples", "cherries", "raspberries", "blackberries", "bananas", "guavas", "lemons", "oranges"]
                selected_objects = random.sample(fruits, k=num_objects)
                object_to_answer_formatter = lambda fruit_name: fruit_name

        object_list = ", ".join(selected_objects[:-1]) + f", and {selected_objects[-1]}"
        intro_str = intro_template.format(num_objects=num2words(num_objects), objects=object_list)

        assignments = list(selected_objects)
        random.shuffle(assignments)

        final_pos = random.choice(list(range(len(selected_objects))))
        true_answer_str = assignments[final_pos]
        if final_pos > len(selected_objects) // 2:
            final_question_str = final_question_template.format(query_position=positions_from_end[final_pos-len(selected_objects)])
        else:
            final_question_str = final_question_template.format(query_position=positions_from_start[final_pos])

        remaining_objects = set(selected_objects) - {true_answer_str}
        order_info = []
        # start with one absolute
        first_absolute_object = random.choice(list(remaining_objects))
        first_absolute_position = assignments.index(first_absolute_object)
        if first_absolute_position > len(selected_objects) // 2:
            order_info.append(absolute_order_template.format(object=first_absolute_object, position=positions_from_end[first_absolute_position-len(selected_objects)]))
        else:
            order_info.append(absolute_order_template.format(object=first_absolute_object, position=positions_from_start[first_absolute_position]))
        remaining_objects.remove(first_absolute_object)
        last_object = first_absolute_object
        
        # add relative orders until the target 
        while len(order_info) < num_steps_to_answer - 1:
            next_object = random.choice(list(remaining_objects))
            pos_diff = assignments.index(next_object) - assignments.index(last_object)
            if pos_diff < 0: 
                direction = directions[0]
            else: direction = directions[1]
            order_info.append(relative_order_template.format(object_1=next_object, object_2=last_object, num_spots=num2words(abs(pos_diff)), direction=direction))

            remaining_objects.remove(next_object)
            last_object = next_object
        
        # now at the target, add last relative order
        pos_diff = final_pos - assignments.index(last_object)
        if pos_diff < 0: 
            direction = directions[0]
        else: direction = directions[1]
        order_info.append(relative_order_template.format(object_1=true_answer_str, object_2=last_object, num_spots=num2words(abs(pos_diff)), direction=direction))
        
        # add miscellaneous information about the rest
        for object_1 in remaining_objects:
            pos_of_object = assignments.index(object_1)
            # maybe absolute or relative
            random_num = random.choice([0,1])
            if random_num == 0:
                if pos_of_object > len(selected_objects) // 2:
                    order_info.append(absolute_order_template.format(object=object_1, position=positions_from_end[pos_of_object-len(selected_objects)]))
                else:
                    order_info.append(absolute_order_template.format(object=object_1, position=positions_from_start[pos_of_object]))
            else:
                random_other_object = random.choice([other_obj for other_obj in selected_objects if other_obj != object_1])
                pos_diff = pos_of_object - assignments.index(random_other_object)
                if pos_diff < 0: 
                    direction = directions[0]
                else: direction = directions[1]
                order_info.append(relative_order_template.format(object_1=object_1, object_2=random_other_object, num_spots=num2words(abs(pos_diff)), direction=direction))
        random.shuffle(order_info)
        sequence_str = " ".join(order_info)
        return intro_str, sequence_str, final_question_str, object_to_answer_formatter(true_answer_str)

    def generate_random(self, generator, kN):
        num_objects = kN['k']
        num_steps = kN['N']

        prompts = []
        true_answers = []
        while len(prompts) < generator.max_batch_size:
            domain_idx = random.choice(range(5))
            intro_str, sequence_str, final_question_str, true_answer = self.generate_random_example(num_objects, num_steps, domain_idx)

            prompt = self.make_prompt(generator, num_objects, intro_str, sequence_str, final_question_str)
            prompts.append(prompt)
            true_answers.append(true_answer)
        return prompts, true_answers

    def make_prompt(self, generator, num_objects, intro_str, sequence_str, final_question_str):
        if 'tokenizer' in dir(generator) and generator.tokenizer.chat_template:
            if "gemma" in generator.model_name:
                messages = [{
                    "role": "user",
                    "content": self.system_instruction + "\n\n" + self.query_template.format(num_objects=num2words(num_objects), intro=intro_str, sequence=sequence_str, final_question=final_question_str) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                messages = [{
                    "role": "system",
                    "content": self.system_instruction
                },
                {
                    "role": "user",
                    "content": self.query_template.format(num_objects=num2words(num_objects), intro=intro_str, sequence=sequence_str, final_question=final_question_str) + "\n" + self.generation_instruction
                }]
                prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.system_instruction + "\n\n"
            prompt += self.query_template.format(num_objects=num2words(num_objects), intro=intro_str, sequence=sequence_str, final_question=final_question_str) + "\n"
            prompt += self.generation_instruction
        return prompt
