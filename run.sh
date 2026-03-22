#!/bin/bash

if [[ $# -ne 1 ]] || [[ "$1" != --version=* ]]; then
    echo "Usage: $0 --version=DIR_NAME"
    echo "Example: $0 --version=myapp"
    exit 1
fi

version_dir="${1#*=}"

cd "versions/$version_dir" || exit 1
exec python3 pipeline.py