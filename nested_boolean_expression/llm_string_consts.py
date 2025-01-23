
system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

query_template = """Evaluate the following boolean expression:

truth_value = {expression}
"""

generation_instruction = "Provide your final answer following this template: [ANSWER]\ntruth_value == YOUR ANSWER\n[/ANSWER]"

reprompt_string = "[ANSWER]\ntruth_value == "
answer_regex = r'\[ANSWER\]\s*truth_value\s*==\s*(True|False|0|1)\s*\[\/ANSWER\]'
stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, expression):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = query_template.format(expression=expression) + "\n"
    prompt += generation_instruction
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, expression, include_starter=False):
    if tokenizer.chat_template: 
        prompt = prompt_with_chat_template(tokenizer, expression)
    else:
        prompt = system_instruction + "\n\n"
        prompt = query_template.format(expression=expression) + "\n"
        prompt += generation_instruction + "\n\n"
    if include_starter:
        prompt += reprompt_string
    return prompt