#!/bin/bash

# Script to convert Jupyter Notebook to Markdown and PDF

NOTEBOOK_PATH="foundational_time_series_blog.ipynb"
OUTPUT_DIR="output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Convert to Markdown
jupyter nbconvert --to markdown $NOTEBOOK_PATH --output-dir $OUTPUT_DIR

# Convert to PDF
jupyter nbconvert --to pdf $NOTEBOOK_PATH --output-dir $OUTPUT_DIR

# Notify user
echo "Conversion complete. Files are saved in the $OUTPUT_DIR directory."