#!/bin/bash

################################################################################
# Dataset Download Script for xBD and BRIGHT
# 
# This script downloads:
# 1. xBD dataset from Kaggle
# 2. BRIGHT dataset from Zenodo
#
# Requirements:
# - kaggle CLI: pip install kaggle
# - kaggle API credentials: ~/.kaggle/kaggle.json
# - wget or curl
# - unzip
#
# Usage:
#   bash scripts/01_download_datasets.sh [--xbd-only | --bright-only]
################################################################################

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
DOWNLOAD_DIR="${DATA_DIR}/downloads"

XBD_DIR="${DATA_DIR}/xbd_raw"
BRIGHT_DIR="${DATA_DIR}/bright_raw"

# Parse arguments
DOWNLOAD_XBD=true
DOWNLOAD_BRIGHT=true

for arg in "$@"; do
    case $arg in
        --xbd-only)
            DOWNLOAD_BRIGHT=false
            ;;
        --bright-only)
            DOWNLOAD_XBD=false
            ;;
        --help)
            echo "Usage: $0 [--xbd-only | --bright-only]"
            echo ""
            echo "Options:"
            echo "  --xbd-only      Download only xBD dataset"
            echo "  --bright-only   Download only BRIGHT dataset"
            echo "  --help          Show this help message"
            exit 0
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_section() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please install it first."
        return 1
    fi
    return 0
}

verify_checksum() {
    local file=$1
    local expected_md5=$2
    
    if [ -f "$file" ]; then
        echo "Verifying checksum for $(basename $file)..."
        local actual_md5=$(md5sum "$file" | awk '{print $1}')
        if [ "$actual_md5" == "$expected_md5" ]; then
            print_success "Checksum verified: $(basename $file)"
            return 0
        else
            print_error "Checksum mismatch for $(basename $file)"
            print_error "Expected: $expected_md5"
            print_error "Got: $actual_md5"
            return 1
        fi
    else
        print_error "File not found: $file"
        return 1
    fi
}

################################################################################
# Setup
################################################################################

print_section "Setting up directories"

mkdir -p "${DATA_DIR}"
mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${XBD_DIR}"
mkdir -p "${BRIGHT_DIR}"

echo "Data directory: ${DATA_DIR}"
echo "Download directory: ${DOWNLOAD_DIR}"

################################################################################
# Download xBD Dataset
################################################################################

if [ "$DOWNLOAD_XBD" = true ]; then
    print_section "Downloading xBD Dataset from Kaggle"
    
    # Check for kaggle CLI
    if ! check_command kaggle; then
        print_error "Please install kaggle CLI: pip install kaggle"
        print_error "And configure credentials: https://github.com/Kaggle/kaggle-api#api-credentials"
        exit 1
    fi
    
    # Check for kaggle credentials
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        print_error "Kaggle credentials not found at ~/.kaggle/kaggle.json"
        print_error "Please download your API token from: https://www.kaggle.com/settings"
        exit 1
    fi
    
    cd "${DOWNLOAD_DIR}"
    
    # Download xBD dataset
    echo "Downloading xBD dataset (this may take a while)..."
    if kaggle datasets download -d qianlanzz/xbd-dataset --force; then
        print_success "xBD dataset downloaded"
    else
        print_error "Failed to download xBD dataset"
        exit 1
    fi
    
    # Extract xBD
    echo "Extracting xBD dataset..."
    if [ -f "xbd-dataset.zip" ]; then
        unzip -q xbd-dataset.zip -d "${XBD_DIR}"
        print_success "xBD dataset extracted to ${XBD_DIR}"
    else
        print_error "xbd-dataset.zip not found"
        exit 1
    fi
    
    cd "${PROJECT_ROOT}"
fi

################################################################################
# Download BRIGHT Dataset
################################################################################

if [ "$DOWNLOAD_BRIGHT" = true ]; then
    print_section "Downloading BRIGHT Dataset from Zenodo"
    
    # Check for wget or curl
    if check_command wget; then
        DOWNLOAD_CMD="wget --continue"
    elif check_command curl; then
        DOWNLOAD_CMD="curl -L -C - -O"
    else
        print_error "Neither wget nor curl is installed"
        exit 1
    fi
    
    cd "${DOWNLOAD_DIR}"
    
    # Zenodo record URL
    ZENODO_RECORD="https://zenodo.org/records/15385983/files"
    
    # Files to download with checksums
    declare -A BRIGHT_FILES=(
        ["pre-event.zip"]="087db04490233e40fd5b53ea1d3b374a"
        ["post-event.zip"]="13dbfff273e95995fee2a868388da4ea"
        ["target.zip"]="d7f48f686e0b01772949c1b5e56e3146"
    )
    
    # Download each file
    for file in "${!BRIGHT_FILES[@]}"; do
        echo ""
        echo "Downloading ${file}..."
        
        if [ -f "${file}" ]; then
            echo "File exists, verifying checksum..."
            if verify_checksum "${file}" "${BRIGHT_FILES[$file]}"; then
                echo "File already downloaded and verified: ${file}"
                continue
            else
                echo "Re-downloading ${file}..."
                rm -f "${file}"
            fi
        fi
        
        # Download
        if [[ $DOWNLOAD_CMD == wget* ]]; then
            wget --continue "${ZENODO_RECORD}/${file}" || {
                print_error "Failed to download ${file}"
                exit 1
            }
        else
            curl -L -C - -O "${ZENODO_RECORD}/${file}" || {
                print_error "Failed to download ${file}"
                exit 1
            }
        fi
        
        # Verify checksum
        if verify_checksum "${file}" "${BRIGHT_FILES[$file]}"; then
            print_success "Downloaded and verified: ${file}"
        else
            print_error "Checksum verification failed for ${file}"
            exit 1
        fi
    done
    
    # Extract BRIGHT files
    print_section "Extracting BRIGHT Dataset"
    
    for file in "${!BRIGHT_FILES[@]}"; do
        echo "Extracting ${file}..."
        if [ -f "${file}" ]; then
            unzip -q "${file}" -d "${BRIGHT_DIR}"
            print_success "Extracted ${file}"
        else
            print_error "${file} not found"
            exit 1
        fi
    done
    
    cd "${PROJECT_ROOT}"
fi

################################################################################
# Summary
################################################################################

print_section "Download Summary"

if [ "$DOWNLOAD_XBD" = true ]; then
    if [ -d "${XBD_DIR}" ]; then
        XBD_SIZE=$(du -sh "${XBD_DIR}" | cut -f1)
        echo "xBD dataset: ${XBD_DIR} (${XBD_SIZE})"
    fi
fi

if [ "$DOWNLOAD_BRIGHT" = true ]; then
    if [ -d "${BRIGHT_DIR}" ]; then
        BRIGHT_SIZE=$(du -sh "${BRIGHT_DIR}" | cut -f1)
        echo "BRIGHT dataset: ${BRIGHT_DIR} (${BRIGHT_SIZE})"
    fi
fi

echo ""
print_success "All downloads completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Run preprocessing script: python scripts/02_preprocess_data.py"
echo "  2. Check the organized data in: ${DATA_DIR}/xbd/ and ${DATA_DIR}/bright/"
echo ""

# Create a download log
LOG_FILE="${DATA_DIR}/download_log.txt"
echo "Download completed: $(date)" > "${LOG_FILE}"
echo "xBD: ${DOWNLOAD_XBD}" >> "${LOG_FILE}"
echo "BRIGHT: ${DOWNLOAD_BRIGHT}" >> "${LOG_FILE}"
if [ "$DOWNLOAD_XBD" = true ]; then
    echo "xBD location: ${XBD_DIR}" >> "${LOG_FILE}"
fi
if [ "$DOWNLOAD_BRIGHT" = true ]; then
    echo "BRIGHT location: ${BRIGHT_DIR}" >> "${LOG_FILE}"
fi
