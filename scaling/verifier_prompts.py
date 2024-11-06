
verification_cot_insn = """You are a smart and helpful AI assistant. You will help with the task of verifying which of a set of outputs is correct, given an input task.
For each task instance, output "yes" or "no" between [CORRECT] and [/CORRECT] tags. Reason step-by-step before arriving at an answer, if it is helpful.

Task instances:
"""

verification_qa_template = """Question: {question}
Answer: {answer}
Verification: {correct_with_tags}"""


verification_code_template = """Code: {question}
Solution: {answer}
Verification: {correct_with_tags}"""