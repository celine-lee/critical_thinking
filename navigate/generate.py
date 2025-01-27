import random
# https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/navigate.json

template = "Take {num_steps} step{multiplier_s} {direction}."

directions = [("forward", "backwards"), ("right", "left"), ("up", "down")]

def generate_random_helper(max_distance_away, target_length, current_length, curr_position, end_at_start):
    """
    Generates a random sequence that either ends at the origin or does not.
    
    Args:
        max_distance_away (int): Size of the world in terms of no. steps away from the origin.
        target_length (int): Target number of turns.
        current_length (int): Current number of steps taken.
        curr_position (List[int]): dimensions (1, 2, or 3)-size tuple describing position before adding this step
        end_at_start (bool): Whether we have to end at the start

    Returns:
        List[str]: List of strings describing the navigation from this step
        bool: ends at start
    """
    if current_length >= target_length:
        return [], all(dim == 0 for dim in curr_position)

    if end_at_start and sum(offset != 0 for offset in curr_position) >= target_length - current_length:
        dim_to_move = random.choice([dim for dim, offset in enumerate(curr_position) if offset != 0])
        amount_to_move = 0 - curr_position[dim_to_move]
    elif end_at_start and (target_length - current_length < 2):
        dim_to_move = random.choice(list(range(len(curr_position))))
        amount_to_move = 0
    else:
        dim_to_move = random.choice(list(range(len(curr_position))))
        if curr_position[dim_to_move] >= max_distance_away:
            amount_to_move = random.choice(list(range(-2 * max_distance_away, 0)))
        elif curr_position[dim_to_move] <= -max_distance_away:
            amount_to_move = random.choice(list(range(0, 2 * max_distance_away)))
        else:
            can_move_left = - (max_distance_away + curr_position[dim_to_move])
            can_move_right = max_distance_away - curr_position[dim_to_move]
            amount_to_move = random.choice(list(range(can_move_left, can_move_right)))

    new_position = list(curr_position)
    new_position[dim_to_move] += amount_to_move
    
    if amount_to_move > 0: direction_idx = 0
    else: direction_idx = 1

    rest_of_navigation, actually_ends_at_start = generate_random_helper(max_distance_away, target_length, current_length + 1, new_position, end_at_start)
    num_steps = amount_to_move if amount_to_move >=0 else -amount_to_move
    multiplier_s = '' if num_steps == 1 else 's'
    return [template.format(num_steps=num_steps, multiplier_s=multiplier_s, direction=directions[dim_to_move][direction_idx])] + rest_of_navigation, actually_ends_at_start

def generate_random(num_dimensions, max_distance_away, target_length):
    end_at_start = random.choice([False, True])
    sequence, actually_ends_at_start = generate_random_helper(max_distance_away, target_length, 0, [0 for _ in range(num_dimensions)], end_at_start)
    return sequence, actually_ends_at_start

