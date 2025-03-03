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
            prompt += self.generation_instruction + "\n\n"
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
            prompt += self.generation_instruction + "\n\n"
        return prompt



class CRUXEvalTask(Task):
    def __init__(self):
        super(CRUXEvalTask, self).__init__("cruxeval_straightlined", r'assert answer\s*==\s*(.+)')
        self.foldername = "cruxeval_straightlined/outputs"
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
            prompt += self.generation_instruction + "\n\n"
        return prompt