#!/bin/bash
# Any subsequent(*) commands which fail will cause the shell script to exit immediately
set -e

# absolute path of this script file
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"

# use 'source' for TA's (they have the 'source' dir), 'release' or `assignment` (for students)
if [ -d "${THIS_DIR}/source" ]; then
    export CODE_DIR="${THIS_DIR}/source"
elif [ -d "${THIS_DIR}/release" ]; then
    export CODE_DIR="${THIS_DIR}/release"
elif [ -d "${THIS_DIR}/assignment" ]; then
    export CODE_DIR="${THIS_DIR}/assignment"
fi

echo "Adding ${CODE_DIR} to PYTHONPATH"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
