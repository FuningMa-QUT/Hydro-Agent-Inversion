# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import interpolate

def get_aquitim_index(components):
    """Get the column index for specific components in 'aqui_tim.dat'."""
    aqui_tim_header = np.loadtxt('.\\aqui_tim.dat', skiprows=8, max_rows=1, dtype=str)
    # Extract header row (skipping the first element)
    aqui_tim_head = aqui_tim_header[1:]  
    components_index = []
    for comp in components:
        index = np.argwhere(aqui_tim_head == comp)
        components_index.append(index)
    
    # Squeeze to handle array dimensionality and return as integer
    components_index = int(np.squeeze(np.array(components_index), axis=None))
    return components_index

def get_condat_index(components):
    """Get the column index for specific components in 'aqui_con.dat'."""
    aqui_con_header = np.loadtxt('.\\aqui_con.dat', skiprows=7, max_rows=1, dtype=str)
    aqui_con_head = aqui_con_header[1:]  
    components_index = []
    for comp in components:
        index = np.argwhere(aqui_con_head == comp)
        components_index.append(index)
    
    components_index = int(np.squeeze(np.array(components_index), axis=None))
    return components_index

def get_aqui_tim_results(components_index, num_elements):
    """Extract time-series results from 'aqui_tim.dat' for specific grid elements."""
    aqui_tim_read = np.loadtxt('.\\aqui_tim.dat', skiprows=11, dtype=str)
    # Convert data to float, skipping coordinate columns
    aqui_tim = aqui_tim_read[:, 2:].astype(float)
    components_tim = aqui_tim[:, components_index]
    
    num_lines = aqui_tim.shape[0]
    line_indices = np.arange(0, num_lines, 1)
    element_components = np.array([])
    
    for i in range(num_elements):
        # Extract indices belonging to the i-th element based on periodicity
        elem_indices = np.argwhere(line_indices % num_elements == i)
        elem_data = components_tim[elem_indices, :]

        if i == 0:
            element_components = elem_data
        else:
            element_components = np.concatenate((element_components, elem_data), axis=1)

    return element_components

def get_time_index_aquitim(name):
    """Find the column index for a specific variable name in 'aqui_tim.dat' header."""
    with open('.\\aqui_tim.dat', 'r') as f:
        lines = f.readlines()
    
    # Header is at line 9 (index 9)
    header_line = lines[9].split()
    for i, col_name in enumerate(header_line):
        if name in col_name:
            return i
    return None

def get_time_results_single(items, skiprows=10):
    """
    Extract time-series data for specified ions.
    items: list of ion IDs (e.g., ['t_ca+2', 't_na+'])
    Returns: Array where Col 0 is Time, and subsequent columns match 'items'.
    """
    time_data = np.loadtxt('.\\aqui_tim.dat', skiprows=skiprows, dtype=str)
    indices = []
    indices.append(get_time_index_aquitim('Time(yr)'))
    
    for item in items:
        indices.append(get_time_index_aquitim(item))
    
    results = time_data[:, indices].astype(np.float32)
    return results

def read_aqui_con(num_time_steps, num_elements):
    """
    Read results from 'aqui_con.dat'.
    num_time_steps: Number of simulation time steps to read.
    num_elements: Number of grid elements.
    Returns: Concatenated results along axis 1.
    """
    skip_rows = 9
    results = None
    
    for i in range(num_time_steps + 1):
        data = np.loadtxt('.\\aqui_con.dat', skiprows=skip_rows, max_rows=num_elements, dtype=float)
        data = np.expand_dims(data, axis=1)
        
        if i == 0:
            results = data
        else:
            results = np.concatenate((results, data), axis=1)
        
        # Move to next time block (+1 for the header line of each block)
        skip_rows += num_elements + 1
        
    return results

def read_aqui_gas(num_time_steps, num_elements):
    """Read results from 'aqui_gas.dat'."""
    skip_rows = 6
    results = None
    
    for i in range(num_time_steps + 1):
        data = np.loadtxt('.\\aqui_gas.dat', skiprows=skip_rows, max_rows=num_elements, dtype=float)
        data = np.expand_dims(data, axis=1)
        
        if i == 0:
            results = data
        else:
            results = np.concatenate((results, data), axis=1)
        
        skip_rows += num_elements + 1
        
    return results

def read_aqui_min(num_time_steps, num_elements):
    """Read mineral results from 'aqui_min.dat'."""
    skip_rows = 9
    results = None
    
    for i in range(num_time_steps + 1):
        data = np.loadtxt('.\\aqui_min.dat', skiprows=skip_rows, max_rows=num_elements, dtype=float)
        data = np.expand_dims(data, axis=1)
        
        if i == 0:
            results = data
        else:
            results = np.concatenate((results, data), axis=1)
            
        skip_rows += num_elements + 1
        
    return results

def get_kdd_tim_results():
    """Read distribution coefficient data from 'kdd_tim.dat'."""
    kdd_df = pd.read_table('kdd_tim.dat', header=4)
    # Clean and split string data manually due to TOUGHREACT formatting
    header_line = ''.join([col for col in kdd_df])
    kdd_data = kdd_df[header_line].str.split(expand=True)
    kdd_array = np.array(kdd_data).astype(float)
    
    # Extract specific result columns
    kdd_results = kdd_array[:, -5:-1]
    t_na = kdd_results[:, 0]
    t_skdd1 = kdd_results[:, 1]
    t_skdd2 = kdd_results[:, 2]
    t_skdd3 = kdd_results[:, 3]
    
    return t_skdd1, t_skdd2, t_skdd3, t_na

def get_observed_concentration(column_pair):
    """
    Extract distance and concentration from Excel monitoring data.
    column_pair: List of [Distance_Column, Concentration_Column] e.g., ['Ca_D', 'Ca']
    """
    df = pd.read_excel('.\\concentration.xlsx')
    col_names = df.columns.values
    
    # Map column names to indices
    idx_dist = np.where(col_names == column_pair[0])[0][0]
    idx_conc = np.where(col_names == column_pair[1])[0][0]
    
    values = np.ones((df.shape[0], 2))
    values[:, 0] = df[col_names[idx_dist]]
    values[:, 1] = df[col_names[idx_conc]]
    
    # Remove NaN rows
    nan_mask = np.isnan(values).any(axis=1)
    clean_values = values[~nan_mask]
    
    return clean_values

def get_aqui_con_index(var_name):
    """Find the column index for a specific variable in 'aqui_con.dat'."""
    with open('.\\aqui_con.dat', 'r') as f:
        lines = f.readlines()
    
    header = lines[8].split(',')
    # Strip whitespace/newlines from components
    header = [h.strip() for h in header if h.strip()]
    
    for i, name in enumerate(header):
        if var_name in name:
            return i
    return None

def get_simulation_data_spatial(aqui_var_name, excel_col_pair, time_step=5, num_elements=17):
    """
    Interpolate simulation results to match sampling distances.
    aqui_var_name: Name in simulation output (e.g., 'pH')
    excel_col_pair: Names in Excel (e.g., ['PH_D', 'PH'])
    """
    aqui_con_data = read_aqui_con(num_time_steps=time_step, num_elements=num_elements)
    
    var_idx = get_aqui_con_index(aqui_var_name)
    dist_idx = get_aqui_con_index('VARIABLES =X')
    
    # Extract data from the last time step
    model_data = aqui_con_data[:, -1, [dist_idx, var_idx]]
    observed_data = get_observed_concentration(excel_col_pair)
    target_distances = observed_data[:, 0]
    
    # Perform quadratic interpolation
    f_interp = interpolate.interp1d(model_data[:, 0], model_data[:, 1], kind='quadratic')
    interpolated_model_values = f_interp(target_distances)
    
    return interpolated_model_values

def get_simulation_data_time(items, item_index, excel_col_pair, interp_option=0):
    """
    Extract time-series simulation data.
    items: List of ion IDs
    item_index: Index of target ion in 'items' list
    interp_option: If 1, interpolates simulation to match observation times.
    """
    observed_data = get_observed_concentration(column_pair=excel_col_pair)
    observed_times = observed_data[:, 0]
    
    time_results = get_time_results_single(items=items)
    # Skip the first initial state if necessary
    time_results = time_results[1:]
    
    if interp_option == 0:
        # Return direct simulation output (excluding time column)
        return time_results[:, item_index + 1]
    else:
        # Interpolate simulation (converting Time from years to hours if required)
        # Note: adjust the multiplier (365*24) based on your specific unit needs
        f_interp = interpolate.interp1d(time_results[:, 0] * 365 * 24, time_results[:, item_index + 1], kind='quadratic')
        interpolated_values = f_interp(observed_times)
        return interpolated_values

def get_mesh_distances(num_elements):
    """Return the spatial coordinates (X) of the mesh grid."""
    initial_results = read_aqui_con(0, num_elements)
    # Extract X coordinates from the first time step
    mesh_coords = initial_results[:, 0, 0]
    return mesh_coords
