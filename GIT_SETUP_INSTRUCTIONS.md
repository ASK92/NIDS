# Git Repository Setup Instructions

## ✅ Local Repository Setup Complete

Your local git repository has been initialized with:
- Username: ASK92
- Email: ask92@duke.edu
- Initial commit created
- Branch renamed to `main`

## 📋 Next Steps: Create Remote Repository and Push

### Option 1: GitHub (Recommended)

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `NIDS`
   - Description: "Network Intrusion Detection System using Machine Learning"
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Add remote and push:**
   ```bash
   git remote add origin https://github.com/ASK92/NIDS.git
   git push -u origin main
   ```

### Option 2: GitLab

1. **Create a new project on GitLab:**
   - Go to https://gitlab.com/projects/new
   - Project name: `NIDS`
   - Visibility: Public or Private
   - **DO NOT** initialize with README
   - Click "Create project"

2. **Add remote and push:**
   ```bash
   git remote add origin https://gitlab.com/ASK92/NIDS.git
   git push -u origin main
   ```

### Option 3: Duke GitLab (if available)

If Duke has its own GitLab instance:
1. Create project named `NIDS` on Duke GitLab
2. Add remote:
   ```bash
   git remote add origin [Duke GitLab URL]/NIDS.git
   git push -u origin main
   ```

## 🔐 Authentication

If you're prompted for credentials:
- **GitHub**: Use a Personal Access Token (not password)
  - Create at: https://github.com/settings/tokens
  - Select `repo` scope
- **GitLab**: Use your GitLab password or access token

## 📝 Quick Command Reference

After creating the remote repository, run these commands:

```bash
# Add remote (replace with your actual URL)
git remote add origin https://github.com/ASK92/NIDS.git

# Verify remote
git remote -v

# Push to remote
git push -u origin main
```

## ✨ Future Updates

To push future changes:
```bash
git add .
git commit -m "Your commit message"
git push
```

