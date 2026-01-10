#!/usr/bin/env python3
"""
Process Bloomberg OMON export (Excel) to standard CSV format.

Usage:
    python process_bloomberg_export.py TLT_OMON_20260109.xlsx
"""

import pandas as pd
import sys
import os
from datetime import datetime
import re


def extract_date_from_filename(filename):
    """Extract date from filename pattern TLT_OMON_YYYYMMDD.xlsx"""
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
    return None


def get_file_timestamp(filepath):
    """Get file modification time"""
    file_stat = os.stat(filepath)
    file_time = datetime.fromtimestamp(file_stat.st_mtime)
    return file_time.strftime('%H:%M')


def process_bloomberg_export(input_file, output_file=None, underlying_price=None):
    """
    Process Bloomberg OMON Excel export to CSV format.

    Args:
        input_file: Path to Excel file
        output_file: Optional output CSV path (defaults to tlt_options_YYYYMMDD.csv)
        underlying_price: Optional underlying price (prompts if not provided)
    """
    # Extract metadata
    quote_date = extract_date_from_filename(input_file)
    quote_time = get_file_timestamp(input_file)

    if not quote_date:
        print(f"Error: Could not extract date from filename {input_file}")
        sys.exit(1)

    print(f"Processing: {input_file}")
    print(f"Quote Date: {quote_date}")
    print(f"Quote Time: {quote_time}")

    # Prompt for underlying price if not provided
    if underlying_price is None:
        while True:
            try:
                underlying_price = float(input("\nEnter the underlying TLT price at quote time: "))
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    else:
        print(f"Underlying Price: {underlying_price}")

    # Read Excel file with headers from row 1 (0-indexed)
    df = pd.read_excel(input_file, header=1)

    # Filter out rows where Strike (column B) is blank
    df = df[df['Strike'].notna()].copy()

    print(f"\nFound {len(df)} valid option records")

    # Define column mappings
    call_columns = {
        'Strike': 'strike',
        'ExDt': 'expiration_date',
        'Bid': 'bid',
        'Ask': 'ask',
        'IVM': 'implied_vol',
        'DM': 'delta',
        'GM': 'gamma',
        'TM': 'theta',
        'VM': 'vega',
        'Volm': 'volume',
        'OInt': 'open_interest'
    }

    put_columns = {
        'Strike.1': 'strike',
        'ExDt.1': 'expiration_date',
        'Bid.1': 'bid',
        'Ask.1': 'ask',
        'IVM.1': 'implied_vol',
        'DM.1': 'delta',
        'GM.1': 'gamma',
        'TM.1': 'theta',
        'VM.1': 'vega',
        'Volm.1': 'volume',
        'OInt.1': 'open_interest'
    }

    # Process calls
    calls = df[list(call_columns.keys())].copy()
    calls = calls.rename(columns=call_columns)
    calls['option_type'] = 'c'

    # Process puts (filter out rows where Strike.1 is blank)
    puts = df[list(put_columns.keys())].copy()
    puts = puts[puts['Strike.1'].notna()].copy()
    puts = puts.rename(columns=put_columns)
    puts['option_type'] = 'p'

    # Combine calls and puts
    combined = pd.concat([calls, puts], ignore_index=True)

    # Add metadata columns
    combined.insert(0, 'quote_date', quote_date)
    combined.insert(1, 'quote_time', quote_time)
    combined.insert(2, 'underlying_price', underlying_price)

    # Convert expiration date format from MM/DD/YY to YYYY-MM-DD
    combined['expiration_date'] = pd.to_datetime(
        combined['expiration_date'],
        format='%m/%d/%y'
    ).dt.strftime('%Y-%m-%d')

    # Reorder columns to match target format
    output_columns = [
        'quote_date', 'quote_time', 'underlying_price', 'option_type',
        'expiration_date', 'strike', 'bid', 'ask', 'implied_vol',
        'delta', 'gamma', 'theta', 'vega', 'volume', 'open_interest'
    ]
    combined = combined[output_columns]

    # Determine output filename
    if output_file is None:
        date_str = quote_date.replace('-', '')
        output_file = f'tlt_options_{date_str}.csv'

    # Save to CSV
    combined.to_csv(output_file, index=False)

    print(f"\nProcessing complete!")
    print(f"Total records: {len(combined)} ({len(calls)} calls + {len(puts)} puts)")
    print(f"Output saved to: {output_file}")

    # Display sample
    print(f"\nFirst 5 records:")
    print(combined.head().to_string())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_bloomberg_export.py <input_excel_file> [output_csv_file] [underlying_price]")
        print("\nExample:")
        print("  python process_bloomberg_export.py TLT_OMON_20260109.xlsx")
        print("  python process_bloomberg_export.py TLT_OMON_20260109.xlsx custom_output.csv 87.10")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].replace('.', '').isdigit() else None

    # Check if last argument is a price
    underlying_price = None
    if len(sys.argv) >= 3:
        try:
            # Try to parse the last argument as a price
            last_arg = sys.argv[-1]
            underlying_price = float(last_arg)
            # If we successfully parsed it and output_file was set to it, clear output_file
            if len(sys.argv) == 3:
                output_file = None
        except ValueError:
            # Last argument is not a price, must be output file
            pass

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    process_bloomberg_export(input_file, output_file, underlying_price)
