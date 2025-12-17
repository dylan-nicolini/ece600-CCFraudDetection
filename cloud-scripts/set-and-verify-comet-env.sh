#!/usr/bin/env bash

echo "Setting Comet ML environment variables..."

# Export environment variables (current shell)
export COMET_API_KEY="iU27xMQWN5Wi4rc3VLC8E34Az"
export COMET_WORKSPACE="dylan-nicolini"
export COMET_PROJECT_NAME="ece600-ccfraud"

echo ""
echo "Environment variables set for THIS SESSION."
echo ""

echo "Verification:"
echo "-----------------------------"
echo "COMET_API_KEY        = $COMET_API_KEY"
echo "COMET_WORKSPACE      = $COMET_WORKSPACE"
echo "COMET_PROJECT_NAME   = $COMET_PROJECT_NAME"
echo "-----------------------------"

echo ""
echo "NOTE:"
echo "• These vars are session-only."
echo "• To make them permanent, see instructions below."
