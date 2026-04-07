#!/bin/bash
# MoB environment setup — idempotent

# Ensure results directory exists
mkdir -p results/experiments_v3

# Verify Python and key dependencies
python -c "import torch; import numpy; import tqdm; print('Dependencies OK')" 2>/dev/null || {
    echo "WARNING: Missing Python dependencies. Run: pip install torch torchvision numpy tqdm matplotlib scipy"
}

echo "MoB environment ready"
