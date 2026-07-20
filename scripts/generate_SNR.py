#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: feynmanliu
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "core"))

from detectability import snr_square

import random
import csv
import numpy as np

def snr_square_random():
    # Randomly select parameters from given ranges
    b = random.uniform(0.07, 0.5)
    p = random.uniform(0.01, 0.32)
    PERIOD = random.uniform(1.3, 130)
    DB_OVER_DT = random.uniform(0.01, 0.03)
    RS_OVER_A = random.uniform(50, 200)
    RS_OVER_A = 1/RS_OVER_A
    NUM_TRANSITS = random.randint(5, 20)
    
    # Call the original snr_square function with these random parameters
    snr_sqr_value = snr_square(b=b, p=p, PERIOD=PERIOD, DB_OVER_DT=DB_OVER_DT, RS_OVER_A=RS_OVER_A, NUM_TRANSITS=NUM_TRANSITS)
    snr_sqr_value = np.sqrt(snr_sqr_value)
    #Save results to CSV
    with open("SNR_random.csv", "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header only once (check if file size is zero)
        if csvfile.tell() == 0:
            csvwriter.writerow(["b", "p", "PERIOD", "DB_OVER_DT", "RS_OVER_A", "NUM_TRANSITS", "snr_sqr_value"])
        # Write the randomly generated parameters and the calculated value
        csvwriter.writerow([b, p, PERIOD, DB_OVER_DT, RS_OVER_A, NUM_TRANSITS, snr_sqr_value])

    
    return snr_sqr_value

#print(snr_square_random())

def filter_and_save_snr():
    # Define the SNR targets and the allowed error
    snr_targets = [3, 5, 10, 20, 30, 50, 100]
    error_margin = 0.01
    
    # Create a dictionary to hold rows for each SNR target
    filtered_rows = {target: [] for target in snr_targets}
    
    # Read the original CSV and filter rows
    with open('../data/SNR_data/SNR_random.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header row
        
        for row in csvreader:
            snr_sqr_value = float(row[-1])  # Assuming snr_sqr_value is the last column
            
            for target in snr_targets:
                lower_bound = target * (1 - error_margin)
                upper_bound = target * (1 + error_margin)
                
                if lower_bound <= snr_sqr_value <= upper_bound:
                    filtered_rows[target].append(row)
    
    # Write the filtered rows to new CSV files
    for target, rows in filtered_rows.items():
        with open(f'SNR_{target}.csv', 'w', newline='') as new_csvfile:
            csvwriter = csv.writer(new_csvfile)
            csvwriter.writerow(header)  # Write the header
            csvwriter.writerows(rows)   # Write the filtered rows

# Example usage:
filter_and_save_snr()











