
######################### ARRAYWORLD #########################


# fs_code = ["""
# x = 33
# y = 148
# z = x + y
# if z % 2:
#     answer = True
# else:
#     answer = False
# assert answer == False
# """, """
# var1 = [3, 5, 7, 2]
# is_sorted = True
# if var1[0] > var[1]: is_sorted = False
# elif var1[1] > var[2]: is_sorted = False
# elif var1[2] > var[3]: is_sorted = False
# answer = is_sorted
# assert answer == False
# """, """
# text, sub = "egmdartoa", "dart"
# answer = sub in text
# assert answer == True
# """,
# ]

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

[ANSWER]assert answer == 2[/ANSWER]

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
assert answer == 4""", 

"""
array = "this sentence has 5 words in it".split()
idx = 5
i = 6
if idx > 5: 
    idx = 0
answer = array[idx]
assert answer == "in"
""", 

"""
array = [9, 99, 999, 99999, 9999]
another_array = [99, 99]
idx = len(another_array)
idx = idx - 3
answer = array[idx]
assert answer == 9999
""",
]

######################### IDX MANAGEMENT #########################


idx_management_cot_query_template = """You are given a snippet of Python code. Complete the assertion with the output of executing the function on the input. First, reason step by step before arriving at an answer. Then, surround the answer as an assertion with [ANSWER] and [/ANSWER] tags.

```
idx = 1
print("hello world")
idx *= 1
assert idx == ??
```

The code manipulates the value of variable `idx`. It first sets it to 1 then multiplies it by 1, to get the final value `1`.

Therefore, the output set to `idx` is 1.

[ANSWER]assert idx == 1[/ANSWER]

```
{code}
```
"""

idx_management_examples = ["""
idx = 0
idx += 4
idx = idx - 1
assert idx == 3""", 
"""
idx = 5
i = 6
if idx > 5: 
    idx = 0
assert idx == 5
""", 
"""
an_array = [99, 99]
idx = len(an_array)
idx = idx - 3
assert idx == -3
""",
]

######################### COMPO GAP #########################

self_ask_prompt = '''Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes. 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
So the final answer is: No

Question: {question}
Are follow up questions needed here:'''


qa_basic_prompt = '''Question: Who lived longer, Muhammad Ali or Alan Turing?
Answer: Muhammad Ali 

Question: When was the founder of craigslist born?
Answer: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Answer: Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Answer: No

Question: {question}
Answer:'''


compgap_cot_prompt = '''Question: Who lived longer, Muhammad Ali or Alan Turing?
Let's think about this step-by-step.
Muhammad Ali was 74 years old when he died.
Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of craigslist born?
Let's think about this step-by-step.
Craigslist was founded by Craig Newmark.
Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Let's think about this step-by-step.
The mother of George Washington was Mary Ball Washington.
The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Let's think about this step-by-step.
The director of Jaws is Steven Spielberg. 
Steven Spielberg is from The United States. 
The director of Casino Royale is Martin Campbell. 
Martin Campbell is from New Zealand. 
So the final answer is: No

Question: {question}
Let's think about this step-by-step.
'''


######################### TRIViA QA #########################



trivia_basic_prompt = '''Question: Who was the man behind The Chipmunks?
Answer: David Seville

Question: Which Lloyd Webber musical premiered in the US on 10th December 1993?
Answer: Sunset Boulevard

Question: Who was the next British Prime Minister after Arthur Balfour?
Answer: Sir Henry Campbell-Bannerman

Question: Who had a 70s No 1 hit with Kiss You All Over?
Answer: Exile

Question: {question}
Answer:'''


trivia_cot_prompt = '''Question: Who was the man behind The Chipmunks?
Let's think about this step-by-step.
The description for the movie The Chipmunks starts: A struggling songwriter named Dave Seville finds success when he comes across a trio of singing chipmunks...
So the final answer is: David Seville

Question: Which Lloyd Webber musical premiered in the US on 10th December 1993?
Let's think about this step-by-step.
Andrew Lloyd Webber's Sunset Boulevard originally premiered at the West End's Adelphi Theatre in 1993.
So the final answer is: Sunset Boulevard

Question: Who was the next British Prime Minister after Arthur Balfour?
Let's think about this step-by-step.
Balfour resigned as Prime Minister in 1905 and was succeeded by Liberal Party politician Sir Henry Campbell-Bannerman.
So the final answer is: Sir Henry Campbell-Bannerman

Question: Who had a 70s No 1 hit with Kiss You All Over?
Let's think about this step-by-step.
"Kiss You All Over" is a 1978 song performed by the group Exile.
So the final answer is: Exile

Question: {question}
Let's think about this step-by-step.
'''


######################### INDEXING #########################

indexing_examples = ["""
array = [3, 4, 63, 1, "hello", 0, 4, 63]
idx = 6
answer = array[idx]
assert answer == 4""", """
array = "this sentence has 5 words in it".split()
idx = 5
answer = array[idx]
assert answer == "in"
""", """
array = [9, 99, 999, 99999, 9999]
idx = -1
answer = array[idx]
assert answer == 9999
""",
]
