import random
import sys
import ipdb
import traceback


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    print()
    ipdb.pm()  # post-mortem debugger


sys.excepthook = debughook

# Generate random DFA with desired number of states, vocabulary, and density.
def random_dfa(N, vocab, d):
    V = len(vocab)

    # get density. sampled from start * symbol to keep it deterministic
    active_edges = random.sample(range(N*V), min(int(d * N * N), N * V))

    # (start, end) -> symbol
    edges = {} 
    # state -> [incoming states, symbol to outgoing state]
    intermediate_node_view = {} 
    # assign random end states
    for edge in active_edges:
        start_state = edge // V
        vocab_elt = vocab[edge % V]
        if start_state not in intermediate_node_view: intermediate_node_view[start_state] = [set(), {}]
        unused_next_state_options = set(range(N)) - set(intermediate_node_view[start_state][-1].values())
        if len(unused_next_state_options) == 0:
            continue
        end_state = random.choice(list(unused_next_state_options))
        edges[(start_state, end_state)] = vocab_elt

        if end_state not in intermediate_node_view: intermediate_node_view[end_state] = [set(), {}]
        intermediate_node_view[end_state][0].add(start_state)
        intermediate_node_view[start_state][1][vocab_elt] = end_state

    N = len(intermediate_node_view)
    intermediate_node_view["start"] = [set(), {"": 0}]
    intermediate_node_view["end"] = [{N - 1}, {}]
    if 0 not in intermediate_node_view: intermediate_node_view[0] = [set(), {}]
    if N-1 not in intermediate_node_view: intermediate_node_view[N-1] = [set(), {}]
    intermediate_node_view[0][0].add("start")
    intermediate_node_view[N-1][1][""] = "end"
    edges[("start", 0)] = ""
    edges[(N - 1, "end")] = ""

    return edges, intermediate_node_view, len(edges) / (N*N)

# Convert DFA to regex
def dfa_to_regex(intermediate_node_view, edges):

    states_to_rip = set(intermediate_node_view.keys()) - {"start", "end"}
    while states_to_rip:

        state_to_rip = states_to_rip.pop()
        incoming_states = intermediate_node_view[state_to_rip][0]
        outgoing_states = intermediate_node_view[state_to_rip][1]

        r_self = f"({edges[(state_to_rip, state_to_rip)]})*" if (state_to_rip, state_to_rip) in edges else ""

        for q_in in incoming_states:
            if q_in == state_to_rip:
                continue
            r_in = edges.get((q_in, state_to_rip), None)
            if r_in is None: breakpoint()

            for q_out in outgoing_states.values():
                if q_out == state_to_rip:
                    continue
                r_out = edges.get((state_to_rip, q_out), None)
                if r_out is None: breakpoint()

                new_r = f"{r_in}{r_self}{r_out}"
                if (q_in, q_out) in edges:
                    new_r = f"({edges[(q_in, q_out)]})|({new_r})"
                edges[(q_in, q_out)] = new_r
                intermediate_node_view[q_out][0].add(q_in)
                intermediate_node_view[q_in][1][new_r] = q_out

        # Remove references to the ripped state
        for q_in in incoming_states:
            r_in = edges[(q_in, state_to_rip)]
            edges.pop((q_in, state_to_rip), None)
            del intermediate_node_view[q_in][1][r_in]
        for q_out in outgoing_states.values():
            edges.pop((state_to_rip, q_out), None)
            intermediate_node_view[q_out][0].remove(state_to_rip)

        intermediate_node_view.pop(state_to_rip)

    return edges[("start", "end")] if ("start", "end") in edges else None


# Get runs
def do_run(intermediate_node_view, edges, max_length=100):
    state = "start"
    generation = ""
    while state != "end":
        if len(intermediate_node_view[state][1]) == 0: return None
        str_addition = random.choice(list(intermediate_node_view[state][1].keys()))
        next_state = intermediate_node_view[state][1][str_addition]
        generation += str_addition
        assert str_addition == edges[(state, next_state)]
        state = next_state
        if len(generation) > max_length: return None
    return generation


def lex(intermediate_node_view, s):
    state = 0
    for c in s:
        next_state = intermediate_node_view[state][1].get(c, None)
        if next_state is None: return False
        state = next_state
    state = intermediate_node_view[state][1].get("", state) 
    return state == "end"

def generate_illegal_string(intermediate_node_view, vocab, max_length, max_tries=100):
    for _ in range(max_tries):
        random_str = "".join(random.choices(vocab, k=random.randint(1, max_length)))
        if not lex(intermediate_node_view, random_str):
            return random_str
    return None

def generate_hard_illegal_string(intermediate_node_view, vocab, degree_wrongness, max_length):
    state = "start"
    generation = ""
    while state != "end":
        # choose an extension, but with some degree of randomness go to the wrong next state
        str_addition = random.choice(list(intermediate_node_view[state][1].keys()))
        next_state = intermediate_node_view[state][1][str_addition]
        if len(intermediate_node_view[state][1]) > 1 and random.randint(0, degree_wrongness*len(intermediate_node_view)) == 0:
            next_state = random.choice(list(set(intermediate_node_view[state][1].values()) - {next_state}))
        generation += str_addition
        state = next_state
        if len(generation) > max_length: break
    return generation, state == "end"
            

# from tqdm import tqdm
# N = 2
# vocab = ["a", "b", "c"]
# d = 0.8
# while True:
#     edges, rip_node_view, actual_d = random_dfa(N, vocab, d)
#     print(edges)
#     runs = []
#     num_repeats_before_quit = 10
#     with tqdm(total=10) as pbar:
#         while len(set(runs)) < 10:
#             run = do_run(rip_node_view, edges)
#             if run is None: break
#             if run not in runs: pbar.update(1)
#             runs.append(run)
#             if runs.count(run) > num_repeats_before_quit: break
#     if run is None: continue
#     resulting_regex = dfa_to_regex(rip_node_view, edges)
#     print(resulting_regex)
#     print(set(runs))
#     if resulting_regex is not None: break