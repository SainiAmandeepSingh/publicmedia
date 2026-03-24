#!/bin/bash
# setup_github.sh
# Run this script ONCE to initialise the local git repo and push to GitHub.
#
# Prerequisites:
# 1. Create a new empty repository on GitHub (no README, no .gitignore)
# 2. Replace YOUR_GITHUB_URL below with your repo URL
# 3. Run: bash setup_github.sh

YOUR_GITHUB_URL="https://github.com/YOUR_USERNAME/npo_recommender.git"

echo "Initialising git repository..."
git init
git add .
git commit -m "Initial project structure: NPO Start Public Values Recommender"

echo "Adding remote origin..."
git remote add origin $YOUR_GITHUB_URL

echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "Done! Your repository is now on GitHub."
echo "Share the URL with your group members."
echo ""
echo "Group members — to get the repo:"
echo "  git clone $YOUR_GITHUB_URL"
echo "  cd npo_recommender"
echo "  python -m venv venv && source venv/bin/activate"
echo "  pip install -r requirements.txt"
