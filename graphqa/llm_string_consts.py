stop_strings = ["[/ANSWER]"]

reprompt_string = "[ANSWER]\nanswer = "


def prompt_with_chat_template(tokenizer, example):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = example["question"]
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, example, include_starter=False):
    if tokenizer.chat_template: 
        prompt = prompt_with_chat_template(tokenizer, example)
    else: 
        prompt = system_instruction + "\n\n"
        prompt += example["question"] + "\n"
    if include_starter:
        prompt += reprompt_string
    return prompt