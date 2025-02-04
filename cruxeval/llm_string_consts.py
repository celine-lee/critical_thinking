
system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

query_template = """You are given a snippet of Python code. Complete the assertion with the output of executing the function on the input.

{fn_def}

answer = f({input_str})
assert answer == ??
"""

generation_instruction = "Provide your final answer following this template: [ANSWER]\nassert answer == YOUR ANSWER\n[/ANSWER]"

reprompt_string = "[ANSWER]\nassert answer == "
answer_regex = r'\[ANSWER\]\s*assert answer\s*==\s*(.+)\s*\[\/ANSWER\]'
stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, fn_def, input_str):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = query_template.format(fn_def=fn_def, input_str=input_str) + "\n"
    prompt += generation_instruction
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, fn_def, input_str, include_starter=False):
    if tokenizer.chat_template: 
        prompt = prompt_with_chat_template(tokenizer, fn_def, input_str)
    else:
        prompt = system_instruction + "\n\n"
        prompt = query_template.format(fn_def=fn_def, input_str=input_str) + "\n"
        prompt += generation_instruction + "\n\n"
    if include_starter:
        prompt += reprompt_string
    return prompt