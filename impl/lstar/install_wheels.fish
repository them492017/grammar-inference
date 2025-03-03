#!/usr/bin/env fish

# Set working directory
set DIR (pwd)

# Create a Python virtual environment
python3 -m venv $DIR/venv

# Install all .whl files in the directory
for whl in $DIR/wheels/*.whl
    venv/bin/python3 -m pip install $whl
end

