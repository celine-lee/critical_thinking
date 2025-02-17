
system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

query_template = """Determine whether the following string belongs to the Dyck language, i.e. is a balanced string of brackets such that every single open bracket has a corresponding closed bracket later in the string.

Input: {dyck_word}
"""

generation_instruction = "Provide your final answer as True or False, following this template: [ANSWER]\nis_balanced == YOUR ANSWER\n[/ANSWER]"

reprompt_string = "[ANSWER]\nis_balanced == "
answer_regex = r'\[ANSWER\]\s*is_balanced\s*==\s*(True|False|0|1)\s*\[\/ANSWER\]'
stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, dyck_word):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = query_template.format(dyck_word=dyck_word) + "\n"
    prompt += generation_instruction
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, dyck_word, include_starter=False):
    if tokenizer.chat_template: 
        prompt = prompt_with_chat_template(tokenizer, dyck_word)
    else:
        prompt = system_instruction + "\n\n"
        prompt = query_template.format(dyck_word=dyck_word) + "\n"
        prompt += generation_instruction + "\n\n"
    if include_starter:
        prompt += reprompt_string
    return prompt