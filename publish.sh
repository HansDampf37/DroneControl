#!/bin/bash
# Script to build and publish drone-rl-env to PyPI
# Usage: ./publish.sh [test|prod]

set -e  # Exit on error

MODE="${1:-test}"

echo "===================================="
echo "Publishing drone-rl-env to PyPI"
echo "Mode: $MODE"
echo "===================================="

# Check if build and twine are installed
if ! python -m build --version &> /dev/null; then
    echo "Error: 'build' not installed. Installing..."
    pip install --upgrade build
fi

if ! python -m twine --version &> /dev/null; then
    echo "Error: 'twine' not installed. Installing..."
    pip install --upgrade twine
fi

# Clean old builds
echo ""
echo "Step 1: Cleaning old builds..."
rm -rf build/ dist/ *.egg-info src/*.egg-info
echo "✓ Cleaned"

# Build package
echo ""
echo "Step 2: Building package..."
python -m build
echo "✓ Built"

# Check package
echo ""
echo "Step 3: Checking package..."
python -m twine check dist/*
echo "✓ Package is valid"

# List distribution files
echo ""
echo "Distribution files created:"
ls -lh dist/
echo ""

# Upload
if [ "$MODE" = "test" ]; then
    echo "Step 4: Uploading to TestPyPI..."
    echo "After upload, test with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ drone-rl-env"
    echo ""
    python -m twine upload --repository testpypi dist/*
    echo ""
    echo "✓ Uploaded to TestPyPI: https://test.pypi.org/project/drone-rl-env/"
elif [ "$MODE" = "prod" ]; then
    echo "Step 4: Uploading to Production PyPI..."
    read -p "Are you sure you want to upload to PRODUCTION PyPI? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        python -m twine upload dist/*
        echo ""
        echo "✓ Uploaded to PyPI: https://pypi.org/project/drone-rl-env/"
        echo ""
        echo "Install with:"
        echo "  pip install drone-rl-env"
    else
        echo "Upload cancelled."
        exit 1
    fi
else
    echo "Error: Invalid mode '$MODE'. Use 'test' or 'prod'."
    exit 1
fi

echo ""
echo "===================================="
echo "Publishing complete!"
echo "===================================="

