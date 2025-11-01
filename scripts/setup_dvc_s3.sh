#!/bin/bash
# Script to configure DVC with S3 remote storage
# This is a simpler version that uses DVC commands directly

set -e

BUCKET_NAME="itesm-mna"
REMOTE_PATH="202502-equipo19"
REGION="us-east-2"
REMOTE_NAME="s3-storage"
USE_AWS_PROFILE=true
AWS_PROFILE="equipo19"

echo "=========================================="
echo "Configuring DVC with S3 Remote Storage"
echo "=========================================="

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
else
    echo "DVC already initialized"
fi

# Remove existing remote if it exists
if dvc remote list 2>/dev/null | grep -q "$REMOTE_NAME"; then
    echo "Removing existing remote '$REMOTE_NAME'..."
    dvc remote remove "$REMOTE_NAME" 2>/dev/null || true
fi

# Add S3 remote
echo "Adding DVC remote '$REMOTE_NAME'..."
dvc remote add -d "$REMOTE_NAME" "s3://${BUCKET_NAME}/${REMOTE_PATH}"

# Configure region
dvc remote modify "$REMOTE_NAME" region "$REGION"

# Configure AWS profile if specified
if [ "$USE_AWS_PROFILE" = true ]; then
    echo "Configuring AWS profile '$AWS_PROFILE'..."
    dvc remote modify "$REMOTE_NAME" profile "$AWS_PROFILE"
    echo "Using AWS profile: $AWS_PROFILE"
else
    echo "Configure credentials manually or set USE_AWS_PROFILE=true"
fi

# Verify configuration
echo ""
echo "=========================================="
echo "Configuration completed!"
echo "=========================================="
echo "Remote name: $REMOTE_NAME"
echo "S3 URL: s3://${BUCKET_NAME}/${REMOTE_PATH}"
echo "Region: $REGION"
echo ""
echo "Current remotes:"
dvc remote list
echo ""
echo "Next steps:"
echo "1. Add data to DVC: dvc add data/raw/"
echo "2. Push to S3: dvc push"
echo "=========================================="

