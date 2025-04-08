#!/bin/bash
set -e

if [ "$EUID" -ne 0 ]; then
    SUDO='sudo'
else
    SUDO=''
fi

echo "🖥️  Checking OS..."
if [[ "$(uname)" == "Linux" ]]; then
    echo "✅ Linux environment detected."
else
    echo "❌ This script is intended for Linux only."
    exit 1
fi

echo ""
echo "📦 Checking for NVIDIA CUDA..."
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc version: $(nvcc --version)"
else
    echo "❌  NVIDIA CUDA not found.  Please install and try again."
    exit 1
fi

echo ""
echo "🔍 Checking for NVIDIA GPU (nvidia-smi)..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected."
    # nvidia-smi
else
    echo "⚠️ nvidia-smi not found. GPU drivers may not be installed or system is CPU-only."
fi

echo ""
echo "📦 Checking for required dependencies..."
packages=(
    curl \
    git \
    python3-dev \
    python3-pip \
    g++ \
    libgeos-dev \
    environment-modules)

for pkg in "${packages[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        missing+=("$pkg")
    fi
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "⚠️ Installing missing dependencies: ${missing[*]}"
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends "${missing[@]}"
else
    echo "✅ All required dependencies are installed."
fi

echo ""
echo "📦 Checking micromamba..."
if command -v micromamba &> /dev/null; then
    echo "✅ micromamba version: $(micromamba --version)"
else
    echo "❌  micromamba not found.  Installing..."
    echo "" | "${SHELL}" <(curl -L "micro.mamba.pm/install.sh")
    export PATH=$PATH:$HOME/.local/bin >> $HOME/.bashrc
    source $HOME/.bashrc
    echo "✅ micromamba $(micromamba --version) installed."
fi


