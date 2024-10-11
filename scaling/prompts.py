


fs_code = ["""
x = 33
y = 148
z = x + y
if z % 2:
    answer = True
else:
    answer = False
assert answer == False
""", """
var1 = [3, 5, 7, 2]
is_sorted = True
if var1[0] > var[1]: is_sorted = False
elif var1[1] > var[2]: is_sorted = False
elif var1[2] > var[3]: is_sorted = False
answer = is_sorted
assert answer == False
""", """
text, sub = "egmdartoa", "dart"
answer = sub in text
assert answer == True
""",
]

code_solving_insn = """Based on the given Python code, complete the assert statement with the output when executing the code. 
Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.

""" 

base_predict_insn = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.


n = 17
answer = n
assert answer == 17


"""


op_insn = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Do NOT output a description for the assert.


n = 17
answer = n
assert answer == 17


"""
op_insn_cot = """You are given a piece of Python code. Complete the assertion with the output of executing the function on the input. First, reason step by step before arriving at an answer. Then, surround the answer as an assertion with [ANSWER] and [/ANSWER] tags.

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

cot_query_template = """You are given a snippet of Python code. Complete the assertion with the output of executing the function on the input. First, reason step by step before arriving at an answer. Then, surround the answer as an assertion with [ANSWER] and [/ANSWER] tags.

```
array = [1, 2, 3]
idx = 1
answer = array[idx]
assert answer == ??
```

The code takes array `array` and indexes into it with index `idx`, then assigns the result to `answer`.
To determine the value of `answer` at the end of the code snippet, we need to `1` index of `array`. Since Python is zero-indexed, the answer is the second element of `array`.

Therefore, the output set to `answer` is 2.

[ANSWER]assert f == 2[/ANSWER]

```
{code}
```
"""

scratchpad_template = """Consider the following program:

```
{code}
```

What is the execution trace?

[BEGIN]

{trace}

[END]"""
scratchpad_query_template = """Consider the following program:

```
{code}
```

What is the execution trace?

[BEGIN]"""

scratchpad_examples = [("""v0 = 6
v0 += 0
v4 = 2
while v4 > 0:
    v4 -= 1
    v0 *= 2
output = v0""","""state: {}
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
state: {"v0": 24, "v4": 0, "output": 24}"""),
("""v0 = 4
v0 -= 0
v0 += 2
v0 -= 0
output = v0""","""state: {}
line: v0 = 4
state: {"v0": 4}
line: v0 -= 0
state: {"v0": 4}
line: v0 += 2
state: {"v0": 6}
line: v0 -= 0
state: {"v0": 6}
line: output = v0
state: {"v0": 6, "output": 6}"""),
("""var1 = [3, 5, 7, 2]
is_sorted = True
if var1[0] > var[1]: is_sorted = False
elif var1[1] > var[2]: is_sorted = False
elif var1[2] > var[3]: is_sorted = False
answer = is_sorted""", """state: {}
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
state: {"var1": [3, 5, 7, 2], "is_sorted": False, "answer": False}"""),
("""text, sub = "egmdartoa", "mart"
answer = sub in text""","""state: {}
line: text, sub = "egmdartoa", "dart"
state: {"text": "egmdartoa", "sub": "dart"}
line: answer = sub in text
state: {"text": "egmdartoa", "sub": "dart", "answer": False}""")
]


arrayworld_generation_insn = """Following the examples given, write another Python code snippet that (1) defines an `array` of size {N} or smaller, (2) defines an `idx`, (3) manipulates `idx`, (4) sets `answer = array[idx]` at the end, and (5) only makes use of the simple built-in Python functions.
Make sure the last line in your code snippet is an assert statement testing the answer.

"""


array_world_examples = ["""
array = [3, 4, 63, 1, "hello", 0, 4, 63]
idx = 0
print("hello world")
idx += 4
if array[idx] == "hello":
    idx = -1
idx = idx - 1
answer = array[idx]
assert answer == 4""", """
array = "this sentence has 5 words in it".split()
idx = 5
i = 6
if idx > 5: 
    idx = 0
answer = array[idx]
assert answer == "in"
""", """
array = [9, 99, 999, 99999, 9999]
another_array = [99, 99]
idx = len(another_array)
idx = idx - 3
answer = array[idx]
assert answer == 9999
""",
]