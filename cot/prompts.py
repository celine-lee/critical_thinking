import random


conditional_probe_str = """
{spaces_in_last}condval = bool({to_evaluate})
{spaces_in_last}print(condval)
{spaces_in_last}exit()
"""
vartype_probe_str = """
{spaces_in_last}varval = eval("{to_evaluate}")
{spaces_in_last}print(f"TYPE:<<{{type(varval).__name__}}>>")
{spaces_in_last}exit()
"""

code_generation_insn = """Following the examples given, write another Python code snippet that makes use of basic Python logic operations to derive an answer. 
Make sure the last line in your code snippet is an assert statement testing the answer.

"""

fs_code = ["""```
x = 33
y = 148
z = x + y
if z % 2:
    answer = True
else:
    answer = False
assert answer == False
```""", """```
var1 = [3, 5, 7, 2]
is_sorted = True
if var1[0] > var[1]: is_sorted = False
elif var1[1] > var[2]: is_sorted = False
elif var1[2] > var[3]: is_sorted = False
answer = is_sorted
assert answer == False
```""", """```
text, sub = "egmdartoa", "dart"
answer = sub in text
assert answer == True
```""",
]

code_solving_prompt = """Based on the given Python code, complete the assert statement with the output when executing the code. 
Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.

""" + '\n\n'.join(fs_code)

base_predict_insn = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.

```
n = 17
answer = n
assert answer == 17
```

```"""


op_insn = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.

```
n = 17
answer = n
assert answer == 17
```

"""
op_insn_cot = """You are given a function and an input. Complete the assertion with the output of executing the function on the input. First, reason step by step before arriving at an answer. Then, surround the answer as an assertion with [ANSWER] and [/ANSWER] tags.

```
s = "hi"
f = s + "a"
assert f == ??
```

The code takes a string s and produces the concatenation of s with the string "a", then assigns the result to f.
To determine the output of executing the code with s set to "hi", we need to concatenate "hi" with "a".

Therefore, the output set to f is "hia".

[ANSWER]assert f == "hia"[/ANSWER]

"""


scratchpad_examples = ["""Consider the following Python code:

```
v0 = 6
v0 += 0
v4 = 2
while v4 > 0:
    v4 -= 1
    v0 *= 2
output = v0
```

What is the execution trace? 

[BEGIN]

state: {}
line: v0 = 6:
state: {"v0": 6}
line: v0 += 0
state: {"v0": 6}
line: v4 = 2
state: {"v0": 6, "v4": 2}
line: while v4 > 0:
state: {"v0": 6, "v4": 2}
line: v4 -= 1
state: {"v0": 6, "v4": 1}
line: v0 *= 2
state: {"v0": 12, "v4": 1}
line: while v4 > 0:
state: {"v0": 12, "v4": 1}
line: v4 -= 1
state: {"v0": 12, "v4": 0}
line: v0 *= 2
state: {"v0": 24, "v4": 0}
line: while v4 > 0:
state: {"v0": 24, "v4": 0}
line: output = v0
state: {"v0": 24, "v4": 0, "output": 24}

[DONE]""",

"""Consider the following Python code:

```
v0 = 4
v0 -= 0
v0 += 2
v0 -= 0
output = v0
```

What is the execution trace?

[BEGIN]

state: {}
line: v0 = 4
state: {"v0": 4}
line: v0 -= 0
state: {"v0": 4}
line: v0 += 2
state: {"v0": 6}
line: v0 -= 0
state: {"v0": 6}
line: output = v0
state: {"v0": 6, "output": 6}

[DONE]""",

"""Consider the following Python code:

```
var1 = [3, 5, 7, 2]
is_sorted = True
if var1[0] > var[1]: is_sorted = False
elif var1[1] > var[2]: is_sorted = False
elif var1[2] > var[3]: is_sorted = False
answer = is_sorted
```

What is the execution trace?

[BEGIN]

state: {}
line: var1 = [3, 5, 7, 2]
state: {"var1": [3, 5, 7, 2]}
line: is_sorted = True
state: {"var1": [3, 5, 7, 2], "is_sorted": True}
line: if var1[0] > var[1]: is_sorted = False
state: {"var1": [3, 5, 7, 2], "is_sorted": True}
line: elif var1[1] > var[2]: is_sorted = False
state: {"var1": [3, 5, 7, 2], "is_sorted": True}
line: elif var1[2] > var[3]: is_sorted = False
state: {"var1": [3, 5, 7, 2], "is_sorted": False}
line: answer = is_sorted
state: {"var1": [3, 5, 7, 2], "is_sorted": False, "answer": False}

[DONE]""",

"""Consider the following Python code:

```
text, sub = "egmdartoa", "mart"
answer = sub in text
```

What is the execution trace?

[BEGIN]

state: {}
line: text, sub = "egmdartoa", "dart"
state: {"text": "egmdartoa", "sub": "dart"}
line: answer = sub in text
state: {"text": "egmdartoa", "sub": "dart", "answer": False}

[DONE]""",
]