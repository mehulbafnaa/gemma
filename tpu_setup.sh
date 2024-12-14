#!/bin/bash
# Comprehensive TPU Environment Setup Script

# Color codes for formatted output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
LOGFILE="tpu_setup_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOGFILE") 2>&1

# Error handling function
handle_error() {
    echo -e "${RED}Error: $1${NC}"
    echo "Setup failed at $(date)"
    exit 1
}

# Success message function
print_success() {
    echo -e "${GREEN}✔ $1${NC}"
}

# Section header function
print_section() {
    echo -e "${YELLOW}============================================"
    echo "$1"
    echo "============================================${NC}"
}

# Trap unexpected errors
set -e
trap 'handle_error "Unexpected error on line $LINENO"' ERR

# Detect Python version (allows flexibility)
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
PYTHON_CMD="python${PYTHON_VERSION}"

# Main setup script
main() {
    print_section "1. System Preparation"
    sudo apt-get update || handle_error "Failed to update package lists"
    sudo apt-get upgrade -y
    sudo apt-get install -y \
        software-properties-common \
        curl \
        gnupg \
        wget \
        git \
        build-essential || handle_error "Failed to install system packages"
    print_success "System packages installed"

    print_section "2. Repository Configuration"
    # Modern repository key management
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/google-cloud.gpg
    echo "deb [signed-by=/etc/apt/keyrings/google-cloud.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
    sudo apt-get update || handle_error "Failed to update after repository configuration"
    print_success "Repositories configured"

    print_section "3. Python and PIP Setup"
    # Ensure pip is installed for current Python
    ${PYTHON_CMD} -m ensurepip --upgrade || handle_error "Failed to install/upgrade pip"
    ${PYTHON_CMD} -m pip install --user --upgrade pip setuptools wheel || handle_error "Pip upgrade failed"
    print_success "Python and pip configured"

    print_section "4. TPU-Specific Python Packages"
    ${PYTHON_CMD} -m pip install --user \
        "jax[tpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \
        jupyter \
        notebook \
        ipykernel \
        numpy \
        pandas \
        torch \
        tensorflow \
        flax \
        optax \
        transformers \
        datasets || handle_error "Failed to install Python packages"
    print_success "TPU-compatible packages installed"

    print_section "5. TPU Jupyter Kernel Setup"
    # TPU-specific Jupyter kernel configuration
    env PYTHON_CONFIGURE_OPTS="--enable-shared" \
        JAX_PLATFORMS=tpu \
        ${PYTHON_CMD} -m ipykernel install --user \
        --name tpu_kernel \
        --display-name "TPU Kernel (Python ${PYTHON_VERSION})" \
        || handle_error "Jupyter kernel setup failed"
    print_success "TPU Jupyter kernel installed"

    print_section "6. Verification"
    # Verify TPU device availability and configuration
    ${PYTHON_CMD} -c "
import jax
import os
print('Python Version:', '$PYTHON_VERSION')
print('JAX Devices:', jax.devices())
print('JAX Platform:', os.environ.get('JAX_PLATFORMS', 'Not set'))
print('Default Backend:', jax.default_backend())
" || handle_error "TPU device verification failed"

    print_section "Setup Complete!"
    echo -e "${GREEN}TPU environment successfully configured at $(date)${NC}"
    echo "Log file: ${LOGFILE}"
    echo "Recommended next steps:"
    echo "1. source ~/.bashrc"
    echo "2. Start Jupyter notebook and select 'TPU Kernel'"
    echo "3. Verify TPU usage in your notebooks"
}

# Execute main function
main
