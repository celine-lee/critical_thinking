import random

def normalize_state(idx, full_wrap):
    num_wrap_actions = 0
    while idx < 0: 
        idx += full_wrap
        num_wrap_actions += 1
    while idx >= full_wrap: 
        idx -= full_wrap
        num_wrap_actions += 1
    return idx, num_wrap_actions

def random_walk(k, m, N):
    turns = []
    curr_state = 0
    num_wraps = 0
    while len(turns) < N:
        transition = m * random.choice(list(range(-k+1, k)))
        if transition < 0:
            turns.append(f"pointer = pointer - {-transition}")
        else:
            turns.append(f"pointer = pointer + {transition}")
        curr_state, wrapped = normalize_state(curr_state + transition, k * m)
        num_wraps += wrapped
    return turns, curr_state % 2 == 0, num_wraps
