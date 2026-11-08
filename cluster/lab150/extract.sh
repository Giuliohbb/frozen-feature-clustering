#!/bin/bash
# Download datasets/weights and cache features. Usage: extract.sh CONFIG
set -euo pipefail
source cluster/lab150/env.sh
activate_venv
CONFIG="$1"

python -c "import torch; assert torch.cuda.is_available(), 'GPU not visible (missing --gres?)'; print('gpu:', torch.cuda.get_device_name(0))"
ffc extract --config "$CONFIG" --data-dir "$FFC_STORE/data" --cache-dir "$FFC_STORE/cache"
