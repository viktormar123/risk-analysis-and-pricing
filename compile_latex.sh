#!/bin/bash

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install LaTeX first."
    echo "Visit: https://www.tug.org/mactex/"
    exit 1
fi

# Check if the tex file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 your_document.tex"
    exit 1
fi

# Get the filename without extension
FILENAME=$(basename -- "$1")
FILENAME="${FILENAME%.*}"

echo "Compiling $1..."

# Run pdflatex twice to resolve references
pdflatex "$1"
pdflatex "$1"

echo "Cleaning up auxiliary files..."
rm -f "$FILENAME.aux" "$FILENAME.log" "$FILENAME.out" "$FILENAME.toc"

echo "Done! Output is in $FILENAME.pdf"

# Open the PDF
echo "Opening $FILENAME.pdf..."
open "$FILENAME.pdf" 