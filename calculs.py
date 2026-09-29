import numpy as np

def calculate_count(values):
    """_summary_

    Args:
        values (_type_): _description_
    """
    count = 0
    
    for value in values:
        if not np.isnan(value):
            count += 1
    return count

def calculate_mean(values):
    """_summary_

    Args:
        values (_type_): _description_
    """
    total = 0
    count = 0
    
    for value in values:
        if not np.isnan(value):
            total += value
            count += 1
            
    return total / count

def calculate_std(values):
    """_summary_

    Args:
        values (_type_): _description_
    """
    mean = calculate_mean(values)
    
    total = 0
    count = 0
    
    for value in values:
        if not np.isnan(value):
            difference = value - mean
            total += difference ** 2
            count += 1
            
    variance = total / count
    
    return np.sqrt(variance)

def calculate_min(values):
    """_summary_

    Args:
        values (_type_): _description_
    """
    minimum = None
    
    for value in values:
        if not np.isnan(value):
            if minimum is None or value < minimum:
                minimum = value
                
    return minimum

def calculate_max(values):
    """_summary_

    Args:
        values (_type_): _description_
    """
    maximum = None
    
    for value in values:
        if not np.isnan(value):
            if maximum is None or value > maximum:
                maximum = value
                
    return maximum

def calculate_percentile(values, percentile):
    """
    Calculate a percentile.
    Args:
        values: Values to analyze
    Returns:
        The requested percentile.
    """
    valid_values = []
    
    for value in values:
        if not np.isnan(value):
            valid_values.append(value)
            
    valid_values.sort()
    position = (len(valid_values) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = lower_index + 1
    
    if upper_index >= len(valid_values):
        return valid_values[lower_index]
    
    lower_value = valid_values[lower_index]
    upper_value = valid_values[upper_index]
    
    fraction = position - lower_index
    
    return lower_value + (upper_value - lower_value) * fraction

def calculate_correlation(value_x, value_y):
    
    mean_x = calculate_mean(value_x)
    mean_y = calculate_mean(value_y)
    numerator = 0
    total_x = 0
    total_y = 0
    for x, y in zip(value_x, value_y):
        difference_x = x - mean_x
        difference_y = y - mean_y
        numerator += difference_x * difference_y
        total_x += difference_x ** 2
        total_y += difference_y ** 2
        
    denominator = np.sqrt(total_x * total_y)
    if denominator == 0:
        return 0
    return numerator / denominator