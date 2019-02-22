'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck 
'''
import itertools


def exhaustive_search(variables, train_model, score_model):
    """ Variable selection using backward elimination 
    
    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores
    
    Returns:
        List of best subset models for increasing number of variables
    """
    # create models of increasing size and determine the best models in each case
    result = []
    for nvariables in range(1, len(variables) + 1):
        best_subset = None
        best_score = None
        best_model = None
        for subset in itertools.combinations(variables, nvariables):
            subset = list(subset)
            subset_model = train_model(subset)
            subset_score = score_model(subset_model, subset)
            if best_subset is None or best_score > subset_score:
                best_subset = subset
                best_score = subset_score
                best_model = subset_model
        result.append({
            'n': nvariables,
            'variables': best_subset,
            'score': best_score,
            'model': best_model,
        })
    return result


def backward_elimination(variables, train_model, score_model, verbose=False):
    """ Variable selection using backward elimination 
    
    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores
    
    Returns:
        (best_model, best_variables) 
    """
    # we start with a model that contains all variables
    best_variables = list(variables)  
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print('Start: score={:.2f}'.format(best_score))
    
    while len(best_variables) > 1:
        step = [(best_score, None, best_model)]
        for removeVar in best_variables:
            step_var = list(best_variables)
            step_var.remove(removeVar)
            step_model = train_model(step_var)
            step_score = score_model(step_model, step_var)
            step.append((step_score, removeVar, step_model))

        # sort by ascending score
        step.sort(key=lambda x: x[0])

        # the first entry is the model with the lowest score
        best_score, removed_step, best_model = step[0]
        if verbose:
            print('Step: score={:.2f}, remove {}'.format(best_score, removed_step))
        if removed_step is None:
            # step here, as removing more variables is detrimental to performance
            break
        best_variables.remove(removed_step)
    return best_model, best_variables


def forward_selection(variables, train_model, score_model, verbose=True):
    """ Variable selection using forward selection 
    
    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores
    
    Returns:
        (best_model, best_variables) 
    """
    # we start with a model that contains no variables
    best_variables = []
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print('Start: score={:.2f}, constant'.format(best_score))
    while True:
        step = [(best_score, None, best_model)]
        for addVar in variables:
            if addVar in best_variables:
                continue
            step_var = list(best_variables)
            step_var.append(addVar)
            step_model = train_model(step_var)
            step_score = score_model(step_model, step_var)
            step.append((step_score, addVar, step_model))
        step.sort(key=lambda x: x[0])

        # the first entry in step is now the model that improved most
        best_score, added_step, best_model = step[0]
        if verbose:
            print('Step: score={:.2f}, add {}'.format(best_score, added_step))
        if added_step is None: 
            # stop here, as adding more variables is detrimental to performance
            break
        best_variables.append(added_step)
    return best_model, best_variables


def stepwise_selection(variables, train_model, score_model, direction='both', verbose=True):
    """ Variable selection using forward and/or backward selection 
    
    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores
        direction: use it to limit stepwise selection to either 'forward' or 'backward'
    
    Returns:
        (best_model, best_variables) 
    """
    FORWARD = 'forward'
    BACKWARD = 'backward'
    directions = [FORWARD, BACKWARD]
    if direction.lower() == FORWARD:
        directions = [FORWARD]
    if direction.lower() == BACKWARD:
        directions = [BACKWARD]
    
    # we start with a model that contains no variables
    best_variables = [] if 'forward' in directions else list(variables)
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print('Start: score={:.2f}, constant'.format(best_score))
    
    while True:
        step = [(best_score, None, best_model, 'unchanged')]
        if FORWARD in directions:
            for variable in variables:
                if variable in best_variables:
                    continue
                step_var = list(best_variables)
                step_var.append(variable)
                step_model = train_model(step_var)
                step_score = score_model(step_model, step_var)
                step.append((step_score, variable, step_model, 'add'))

        if 'backward' in directions:
            for variable in best_variables:
                step_var = list(best_variables)
                step_var.remove(variable)
                step_model = train_model(step_var)
                step_score = score_model(step_model, step_var)
                step.append((step_score, variable, step_model, 'remove'))

        # sort by ascending score
        step.sort(key=lambda x: x[0])

        # the first entry is the model with the lowest score
        best_score, chosen_variable, best_model, direction = step[0]
        if verbose:
            print('Step: score={:.2f}, {} {}'.format(best_score, direction, chosen_variable))
        if chosen_variable is None:
            # step here, as adding or removing more variables is detrimental to performance
            break
        if direction == 'add':
            best_variables.append(chosen_variable)
        else:
            best_variables.remove(chosen_variable)   
    return best_model, best_variables
