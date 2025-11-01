#!/bin/bash
# Cross-platform LaTeX Beamer Presentation Compiler
# Usage: ./compile_presentation.sh <latex_file_name>
# Example: ./compile_presentation.sh esg_presentation.tex

# Check if parameter is provided
if [ -z "$1" ]; then
    echo "Error: No LaTeX file specified"
    echo "Usage: ./compile_presentation.sh <latex_file_name>"
    echo "Example: ./compile_presentation.sh esg_presentation.tex"
    exit 1
fi

# Get the filename and remove .tex extension if present
LATEX_FILE="$1"
BASENAME="${LATEX_FILE%.tex}"

# Check if file exists
if [ ! -f "$BASENAME.tex" ]; then
    echo "Error: File '$BASENAME.tex' not found!"
    exit 1
fi

echo "=========================================="
echo "Compiling LaTeX Presentation: $BASENAME.tex"
echo "=========================================="

# First pass
echo "Running first compilation pass..."
pdflatex -interaction=nonstopmode "$BASENAME.tex" > /dev/null 2>&1

# Check if PDF was created (not just exit code, as warnings can cause non-zero exit)
if [ ! -f "$BASENAME.pdf" ]; then
    echo "❌ First compilation pass failed - no PDF created!"
    echo "Check $BASENAME.log for errors"
    exit 1
fi

# Second pass (for TOC and references)
echo "Running second compilation pass..."
pdflatex -interaction=nonstopmode "$BASENAME.tex" > /dev/null 2>&1

# Check if PDF still exists and was updated
if [ ! -f "$BASENAME.pdf" ]; then
    echo "❌ Second compilation pass failed!"
    echo "Check $BASENAME.log for errors"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Compilation complete!"
echo "✓ Output: $BASENAME.pdf"
echo "=========================================="

# Optional: Clean up auxiliary files
# Uncomment the following lines if you want to remove auxiliary files
# echo "Cleaning up auxiliary files..."
# rm -f "$BASENAME.aux" "$BASENAME.log" "$BASENAME.nav" "$BASENAME.out" \
#       "$BASENAME.snm" "$BASENAME.toc" "$BASENAME.vrb"

# Open the PDF based on the operating system
echo ""
echo "Attempting to open PDF..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "$BASENAME.pdf"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open "$BASENAME.pdf"
    elif command -v evince &> /dev/null; then
        evince "$BASENAME.pdf"
    elif command -v okular &> /dev/null; then
        okular "$BASENAME.pdf"
    else
        echo "No PDF viewer found. Please open $BASENAME.pdf manually."
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Git Bash or Cygwin on Windows
    start "$BASENAME.pdf"
else
    echo "Unable to detect OS. Please open $BASENAME.pdf manually."
fi
