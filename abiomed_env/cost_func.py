import numpy as np


MAP_IDX = 0
HR_IDX = 9
PULSATILITY_IDX = 7

arbitrary_threshold_map = -1.36
arbitrary_threshold_hr = -2.16
arbitrary_threshold_pulsatility = -1.95


def is_stable(states):
    """
    Checks if a 1 hour window which is 6 steps is stable.

    Args:
        states (list[list[float]]): A list with exactly 6 state
            vectors with the different features

    Returns:
        bool: Returns True if the hour is stable and False if not
    """
    # assert len(states) == 6, "There are not 6 timesteps"
    hour_np = np.array(states)
    map_values = hour_np[..., MAP_IDX]
    hr_values = hour_np[..., HR_IDX]
    pulsatility_values = hour_np[..., PULSATILITY_IDX]

    is_map_unstable = min(map_values) < 60.0
    is_hr_unstable = (min(hr_values) < 50.0) # or (max(hr_values) >= 100.0)
    is_pulsatility_unstable = min(pulsatility_values) < 20.0

    if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
        return False
    return True


def compute_acp_cost(actions, states):
    """Calculates the Action Change Penalty (ACP) for a single episode

    This function iterates through a sequence of actions in an episode and
    sums the change between each consecutive action

    Args:
        actions (list[float] or np.ndarray) is a 1D list or array of
            actions within a single episode

    Returns:
        float: The cumulative action change penalty for the episode
    """
    reshaped_states = states.reshape(-1, 6, 12)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(reshaped_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    acp = 0.0
    for i in range(1, len(all_actions)):
        if np.linalg.norm((all_actions[i] - all_actions[i-1])) > 2:
            acp += np.linalg.norm((all_actions[i] - all_actions[i-1]))
    return acp


def compute_acp_cost_model(world_model, actions, states):
    """Calculates the Action Change Penalty (ACP) for a single episode

    This function iterates through a sequence of actions in an episode and
    sums the change between each consecutive action

    Args:
        actions (list[float] or np.ndarray) is a 1D list or array of
            actions within a single episode

    Returns:
        float: The cumulative action change penalty for the episode
    """
    reshaped_states = states.reshape(-1, world_model.forecast_horizon, 12)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(unnormalized_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    acp = 0.0
    for i in range(1, len(all_actions)):
        if np.linalg.norm((all_actions[i] - all_actions[i-1])) > 2:
            acp += np.linalg.norm((all_actions[i] - all_actions[i-1]))
    return acp


def is_stable_gradient(states):
    """
    Checks if a 1 hour window which is 6 steps is stable using the definition that 
    MAP, HR, Pulsatility gradients are not below a threshold

    Args:
        states (list[list[float]]): A list with exactly 6 state
            vectors with the different features

    Returns:
        bool: Returns True if the hour is stable and False if not
    """
    states_np = np.array(states)
    x_vals = np.arange(len(states_np))

    map_values = states_np[:, MAP_IDX]
    hr_values = states_np[:, HR_IDX]
    pulsatility_values = states_np[:, PULSATILITY_IDX]

    map_slope = np.polyfit(x_vals, map_values, 1)[0]
    hr_slope = np.polyfit(x_vals, hr_values, 1)[0]
    pulsatility_slope = np.polyfit(x_vals, pulsatility_values, 1)[0]
    # print(pulsatility_slope)
    is_map_unstable = abs(map_slope) >= -arbitrary_threshold_map
    is_hr_unstable = abs(hr_slope) >= -arbitrary_threshold_hr
    is_pulsatility_unstable = abs(pulsatility_slope) >= -arbitrary_threshold_pulsatility

    if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
        return (False,  [map_slope, hr_slope, pulsatility_slope])
    return (True, [map_slope, hr_slope, pulsatility_slope])


def unstable_percentage(flattened_states):
    """
    Calculates the percentage of total timesteps that are in an unstable state, 
        which for now is when MAP, HR, or pulsatility are out of the proper range

    Args:
        flattened_states (list[list[float]]): A 2D array where each row is a state
                                     vector for a single timestep.

    Returns:
        percentage (float) is the percentage of unsafe states
    """
    unstable_hour_count = 0
    total_hours = 0
    for i in range(0, len(flattened_states) - 5, 6):
        total_hours += 1
        hour_chunk_np = np.array(flattened_states[i : i+6])
        map_in_hour = hour_chunk_np[:, MAP_IDX]
        hr_in_hour = hour_chunk_np[:, HR_IDX]
        pulsatility_in_hour = hour_chunk_np[:, PULSATILITY_IDX]

        is_map_unstable = (map_in_hour < 60).any() 
        is_hr_unstable = (hr_in_hour < 50).any()
        is_pulsatility_unstable = (pulsatility_in_hour < 20).any()


        if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
            unstable_hour_count += 1
    if total_hours == 0:
        return 0.0
            
    percentage = (unstable_hour_count / total_hours) * 100
    return percentage


def unstable_percentage_model(world_model, states):
    """
    Calculates the percentage of total timesteps that are in an unstable state, 
        which for now is when MAP, HR, or pulsatility are out of the proper range

    Args:
        states (list[list[float]]): A 2D array where each row is a state
                                     vector for a single timestep.

    Returns:
        percentage (float) is the percentage of unsafe states
    """
    unstable_hour_count = 0
    total_hours = 0
    reshaped_states = states.reshape(len(states), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_state_vectors(reshaped_states)
    for i in range(0, len(unnormalized_states)):
        total_hours += 1
        current_hour_data = unnormalized_states[i]
        map_in_hour = current_hour_data[:,MAP_IDX]
        hr_in_hour = current_hour_data[:,HR_IDX]
        pulsatility_in_hour = current_hour_data[:,PULSATILITY_IDX]

        is_map_unstable = (map_in_hour < 60).any() 
        is_hr_unstable = (hr_in_hour < 50).any()
        is_pulsatility_unstable = (pulsatility_in_hour < 20).any()


        if is_map_unstable or is_hr_unstable or is_pulsatility_unstable:
            unstable_hour_count += 1
    if total_hours == 0:
        return 0.0
            
    percentage = (unstable_hour_count / total_hours) * 100
    return percentage

def unstable_percentage_model_gradient(world_model, states):
    """
    Calculates the percentage of total timesteps that are in an unstable state, 
        which for now is when MAP, HR, or pulsatility are out of the proper range

    Args:
        states (list[list[float]]): A 2D array where each row is a state
                                     vector for a single timestep.

    Returns:
        percentage (float) is the percentage of unsafe states
    """
    unstable_hour_count = 0
    total_hours = 0
    reshaped_states = states.reshape(len(states), world_model.forecast_horizon, -1)
    unnormalized_states = world_model.unnorm_state_vectors(reshaped_states)
    for i in range(0, len(unnormalized_states)):
        total_hours += 1
        current_hour_data = unnormalized_states[i]
        stability = is_stable_gradient(current_hour_data)[0]
        if not stability:
            unstable_hour_count += 1
    if total_hours == 0:
        return 0.0
            
    percentage = (unstable_hour_count / total_hours) * 100
    return percentage


def weaning_score_physician(flattened_states, actions):
    """
    Calculates a weaning score from hourly states and actions. Lowering p level by one is proper 
    weaning and increasing while stable is improper (so it is proportionally penalized)

    Args:
        flattened_states (list[list[float]]): A 2D array of unnormalized
            state vectors for an entire time series
        actions (list[float]): A list of p levels for each state.

    Returns:
        float: The average weaning score per stable hour. A higher score
               means better weaning decisions, but we expect lower values.
    """
   
    reshaped_states = flattened_states.reshape(-1, 6, 12)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(reshaped_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    score = 0.0
    denom = 0.0
    for t in range(1, len(all_actions)):
        if is_stable(flattened_states[t-1]):
            denom += 1.0
            current_action = all_actions[t]
            previous_action = all_actions[t-1]
            increase_diff = current_action - previous_action
            if ((previous_action-current_action) == 1) or ((previous_action-current_action) == 2) :
                score += previous_action-current_action
            
            elif increase_diff > 0:
                score -= 1

    return score / denom if denom != 0 else 0.0


def weaning_score_model(world_model, states, actions):
    """
    Calculates a weaning score from hourly states and actions. Lowering p level by one is proper 
    weaning and increasing while stable is improper (so it is proportionally penalized)

    Args:
        flattened_states (list[list[float]]): A 2D array of unnormalized
            state vectors for an entire time series
        action (list[float]): A list of p levels for each state, unnormalized.

    Returns:
        float: The average weaning score per stable hour. A higher score
               means better weaning decisions, but we expect lower values.
    """

    reshaped_states = states.reshape(-1, world_model.forecast_horizon, 12)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(unnormalized_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    score = 0.0
    denom = 0.0
    for t in range(1, len(all_actions)):
        if is_stable(unnormalized_states[t-1]):
            denom += 1.0
            current_action = all_actions[t]
            previous_action = all_actions[t-1]
            increase_diff = current_action - previous_action
            if ((previous_action-current_action) == 1) or ((previous_action-current_action) == 2) :
                score += (previous_action-current_action)
            
            elif increase_diff > 0:
                score -= 1

    return score / denom if denom != 0 else 0.0


def weaning_score_physician_gradient(flattened_states, actions):
    """
    Calculates a weaning score from hourly states and actions. Lowering p level by one is proper 
    weaning and increasing while stable is improper (so it is proportionally penalized)

    Args:
        flattened_states (list[list[float]]): A 2D array of unnormalized
            state vectors for an entire time series
        actions (list[float]): A list of p levels for each state.

    Returns:
        float: The average weaning score per stable hour. A higher score
               means better weaning decisions, but we expect lower values.
    """
    reshaped_states = flattened_states.reshape(-1, 6, 12)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(reshaped_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    score = 0.0
    denom = 0.0
    slopes = []
    for t in range(1, len(all_actions)):
        if is_stable_gradient(flattened_states[t-1])[0]:
            denom += 1.0
            current_action = all_actions[t]
            previous_action = all_actions[t-1]
            increase_diff = current_action - previous_action
            if ((previous_action-current_action) == 1) or ((previous_action-current_action) == 2) :
                score += previous_action-current_action
            
            elif increase_diff > 0:
                score -= 1
        slopes.append(is_stable_gradient(flattened_states[t-1])[1])
    return score / denom if denom != 0 else 0.0, slopes


def weaning_score_model_gradient(world_model, states, actions):
    """
    Calculates a weaning score from hourly states and actions. Lowering p level by one is proper 
    weaning and increasing while stable is improper (so it is proportionally penalized)

    Args:
        flattened_states (list[list[float]]): A 2D array of unnormalized
            state vectors for an entire time series
        action (list[float]): A list of p levels for each state, unnormalized.

    Returns:
        float: The average weaning score per stable hour. A higher score
               means better weaning decisions, but we expect lower values.
    """

    reshaped_states = states.reshape(-1, world_model.forecast_horizon, 12)
    unnormalized_states = world_model.unnorm_output(reshaped_states)
    first_action_unnorm = np.array(np.bincount(np.rint(np.array(unnormalized_states[0,:,-1])).astype(int)).argmax()).reshape(-1)
    all_actions = np.concatenate([first_action_unnorm, np.asarray(actions, dtype=float)])
    score = 0.0
    denom = 0.0
    slopes = []
    for t in range(1, len(all_actions)):
        stable, slope = is_stable_gradient(unnormalized_states[t-1])
        if stable:
            denom += 1.0
            current_action = all_actions[t]
            previous_action = all_actions[t-1]
            increase_diff = current_action - previous_action
            if ((previous_action-current_action) == 1) or ((previous_action-current_action) == 2) :
                score += (previous_action-current_action)
            
            elif increase_diff > 0:
                score -= 1
        slopes.append(slope)

    return score / denom if denom != 0 else 0.0, slopes