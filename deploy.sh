#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build and deploy Thomas Chang's site to GitHub Pages
#
# Usage:
#   ./deploy.sh              # Deploy with a default commit message
#   ./deploy.sh "My message" # Deploy with a custom commit message
#
# How it works:
#   1. Builds the Hugo site into sites/thomas_chang/public/
#   2. Copies the built files into a temporary worktree on the gh-pages branch
#   3. Pushes gh-pages to GitHub, which GitHub Pages serves as the live site
#
# One-time setup required:
#   In your GitHub repo → Settings → Pages → Source: set to "Deploy from branch"
#   and choose the "gh-pages" branch and "/" (root) folder.
#
# Monitoring & Debugging:
#   Live site:
#     https://thomaschangsf.github.io/
#
#   GitHub Pages deployment status (shows if Pages is building/deployed/failed):
#     https://github.com/thomaschangsf/thomaschangsf.github.io/deployments
#
#   GitHub Actions runs (if using Actions-based deployment):
#     https://github.com/thomaschangsf/thomaschangsf.github.io/actions
#
#   gh-pages branch contents (what is actually being served):
#     https://github.com/thomaschangsf/thomaschangsf.github.io/tree/gh-pages
#
#   Pages settings (source branch, custom domain, HTTPS):
#     https://github.com/thomaschangsf/thomaschangsf.github.io/settings/pages
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$REPO_ROOT/sites/thomas_chang"
BUILD_DIR="$SITE_DIR/public"
DEPLOY_BRANCH="gh-pages"
REMOTE="origin"
BASE_URL="https://thomaschangsf.github.io/"
COMMIT_MSG="${1:-"Deploy site: $(date '+%Y-%m-%d %H:%M:%S')"}"

echo "========================================"
echo "  Thomas Chang Site Deployer"
echo "========================================"

# ── 1. Build the site ─────────────────────────────────────────────────────────
echo ""
echo "▶ Building Hugo site..."
cd "$SITE_DIR"
hugo --gc --minify --baseURL "$BASE_URL"
echo "  ✓ Build complete → $BUILD_DIR"

# ── 2. Ensure gh-pages branch exists on remote ───────────────────────────────
cd "$REPO_ROOT"
echo ""
echo "▶ Preparing $DEPLOY_BRANCH branch..."

# Check if gh-pages branch exists remotely; create it if not
if ! git ls-remote --exit-code --heads "$REMOTE" "$DEPLOY_BRANCH" > /dev/null 2>&1; then
  echo "  Creating orphan $DEPLOY_BRANCH branch..."
  git checkout --orphan "$DEPLOY_BRANCH"
  git reset --hard
  git commit --allow-empty -m "Initial gh-pages branch"
# Return to main branch (clean up lock file first)
rm -f "$SITE_DIR/.hugo_build.lock"
git checkout main
fi

# ── 3. Use a git worktree to safely deploy ───────────────────────────────────
WORKTREE_DIR="$(mktemp -d)"
trap 'git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true; rm -rf "$WORKTREE_DIR"' EXIT

echo "  ✓ Checking out $DEPLOY_BRANCH into temp worktree..."
git fetch "$REMOTE" "$DEPLOY_BRANCH"
git worktree add "$WORKTREE_DIR" "$DEPLOY_BRANCH"

# ── 4. Sync built files into the worktree ────────────────────────────────────
echo ""
echo "▶ Syncing built files..."
# Clear old content but preserve .git
find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# Copy fresh build
cp -r "$BUILD_DIR/." "$WORKTREE_DIR/"

# Add a .nojekyll file so GitHub doesn't try to process with Jekyll
touch "$WORKTREE_DIR/.nojekyll"

echo "  ✓ Files synced"

# ── 5. Commit and push ────────────────────────────────────────────────────────
echo ""
echo "▶ Committing and pushing to $DEPLOY_BRANCH..."
cd "$WORKTREE_DIR"
git add -A

if git diff --cached --quiet; then
  echo "  ℹ  No changes to deploy — site is already up to date."
else
  git commit -m "$COMMIT_MSG"
  git push "$REMOTE" "$DEPLOY_BRANCH"
  echo "  ✓ Deployed successfully!"
  echo ""
  echo "  🌐 Live at:       $BASE_URL"
  echo "     (Changes may take 1-2 minutes to appear)"
  echo ""
  echo "  🔍 Monitor & Debug:"
  echo "     Deployment status: https://github.com/thomaschangsf/thomaschangsf.github.io/deployments"
  echo "     gh-pages contents: https://github.com/thomaschangsf/thomaschangsf.github.io/tree/gh-pages"
  echo "     Pages settings:    https://github.com/thomaschangsf/thomaschangsf.github.io/settings/pages"
fi

echo ""
echo "========================================"
echo "  Done!"
echo "========================================"
