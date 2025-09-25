#!/bin/bash

# Add all changes
git add .

# Generate a random commit message
RANDOM_MSG="commit-$(date +%s)-$RANDOM"

# Commit with the random message
git commit -m "$RANDOM_MSG"

# Push to the current branch on origin
git push origin
