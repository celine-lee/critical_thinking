
system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

query_template = """If you follow these instructions, do you return to the starting point? Always face forward. 

{sequence}
"""

generation_instruction = "Provide your final answer as True or False, following this template: [ANSWER]\nreturned_to_start == YOUR ANSWER\n[/ANSWER]"

reprompt_string = "[ANSWER]\nreturned_to_start == "
answer_regex = r'\[ANSWER\]\s*returned_to_start\s*==\s*(True|False|0|1)\s*\[\/ANSWER\]'
stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, sequence):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = query_template.format(sequence=sequence) + "\n"
    prompt += generation_instruction
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, sequence, include_starter=False):
    if tokenizer.chat_template: 
        prompt = prompt_with_chat_template(tokenizer, sequence)
    else:
        prompt = system_instruction + "\n\n"
        prompt = query_template.format(sequence=sequence) + "\n"
        prompt += generation_instruction + "\n\n"
    if include_starter:
        prompt += reprompt_string
    return prompt