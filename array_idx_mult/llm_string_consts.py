
system_instruction = """You are a smart and helpful AI assistant. Please help me with the following task."""

query_template = """You are given a length-{full_size} array and must track the index of a 0-indexed pointer to the array. The pointer undergoes several modifications. The pointer wraps around the length of the array on both ends, so when it reaches {full_size} it becomes 0, when it reaches {full_size_plus_1} it becomes 1, when it reaches -1 it becomes {full_size_minus_1}, etc. What is the index of the pointer after all the modifications are complete? Provide the answer in the range [0, {full_size}).

pointer = 0
{sequence}
"""

generation_instruction = "Provide your final answer following this template: [ANSWER]\npointer == YOUR ANSWER\n[/ANSWER]"

stop_strings = ["[/ANSWER]"]

def prompt_with_chat_template(tokenizer, k, m, turns):
    messages = []
    if "gemma" not in tokenizer.name_or_path:
        messages.append({
            "role": "system",
            "content": system_instruction
        }
        )
    prompt = query_template.format(full_size=k*m, full_size_plus_1=(k*m)+1, full_size_minus_1=(k*m)-1, sequence="\n".join(turns)) + "\n"
    prompt += generation_instruction
    messages.append({
        "role": "user",
        "content": prompt if "gemma" not in tokenizer.name_or_path else system_instruction + "\n\n" + prompt
    })
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def make_prompt(tokenizer, k, m, turns):
    if tokenizer.chat_template: return prompt_with_chat_template(tokenizer, k, m, turns)
    prompt = system_instruction + "\n\n"
    prompt += query_template.format(full_size=k*m, full_size_plus_1=(k*m)+1, full_size_minus_1=(k*m)-1, sequence="\n".join(turns)) + "\n"
    prompt += generation_instruction + "\n\n"
    return prompt