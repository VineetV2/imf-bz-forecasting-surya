# GitHub Upload Instructions

## ✅ What's Already Done

1. ✓ Git repository initialized
2. ✓ .gitignore created (excludes large files: data, checkpoints, logs)
3. ✓ README.md updated with complete project documentation
4. ✓ Progress email drafted (progress_email_final.txt)
5. ✓ Initial commit created with 22 files

## 📦 What's Included in Repository

**Included** (22 files, ~4,443 lines):
- ✓ Code files (Python scripts)
- ✓ Configuration files (YAML)
- ✓ Model definitions (models/)
- ✓ Utilities (utils/)
- ✓ Result summaries (lofo_results/*/lofo_results.csv)
- ✓ Documentation (README.md, progress_email_final.txt)
- ✓ Requirements (requirements.txt)
- ✓ Git ignore rules (.gitignore)

**Excluded** (via .gitignore):
- ✗ Large data files (data/ ~12GB)
- ✗ Model checkpoints (checkpoints/ ~11GB)
- ✗ LOFO fold checkpoints (lofo_results/*/fold_*/ ~103GB)
- ✗ Training logs (*.log files)
- ✗ Temporary files

**Total repository size**: ~5-10 MB (safe for GitHub)

---

## 🚀 Upload to GitHub - Step by Step

### Option 1: Create New Repository on GitHub (Recommended)

#### Step 1: Create Repository on GitHub Website

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in details:
   - **Repository name**: `imf-bz-forecasting-surya` (or your preferred name)
   - **Description**: `Transfer learning from Surya foundation model for multi-day IMF Bz forecasting using LOFO cross-validation`
   - **Visibility**:
     - ✅ **Public** (recommended for research visibility)
     - OR Private (if you prefer)
   - **Initialize repository**:
     - ❌ **DO NOT** check "Add README" (we already have one)
     - ❌ **DO NOT** check "Add .gitignore"
     - ❌ **DO NOT** check "Choose a license"
4. Click **"Create repository"**

#### Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Navigate to your project directory
cd /Users/vora/Documents/Surya-main/bz_forecasting_complete

# Add GitHub as remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/imf-bz-forecasting-surya.git

# Rename branch to 'main' (GitHub's default)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example** (replace `yourusername` with your actual GitHub username):
```bash
git remote add origin https://github.com/yourusername/imf-bz-forecasting-surya.git
git branch -M main
git push -u origin main
```

#### Step 3: Verify Upload

1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/imf-bz-forecasting-surya`
2. You should see:
   - README.md displayed automatically
   - 22 files
   - Initial commit message
   - Repository size: ~5-10 MB

---

### Option 2: Push to Existing Repository

If you already have a repository:

```bash
cd /Users/vora/Documents/Surya-main/bz_forecasting_complete

# Add your existing repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to main branch
git branch -M main
git push -u origin main
```

---

## 🔐 Authentication

GitHub may ask for authentication when you push. You have two options:

### Option A: Personal Access Token (Recommended)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name: "Laptop Git Access"
4. Select scopes: ✓ repo (all repo permissions)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. When pushing, use token as password:
   - Username: your GitHub username
   - Password: paste the token

### Option B: GitHub CLI

```bash
# Install GitHub CLI
brew install gh

# Authenticate
gh auth login

# Then push normally
git push -u origin main
```

### Option C: SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Paste the copied key

# Change remote URL to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/imf-bz-forecasting-surya.git

# Push
git push -u origin main
```

---

## 📝 After Uploading

### 1. Add a License

GitHub will prompt you to add a license. Common choices:
- **MIT License**: Permissive, allows commercial use
- **Apache 2.0**: Permissive with patent protection
- **GPL-3.0**: Copyleft, derivatives must be open-source

**To add**:
1. Click "Add file" → "Create new file"
2. Name it: `LICENSE`
3. GitHub will offer a template selector
4. Choose license → Commit

### 2. Update Email and Contact Info

Edit `README.md` on GitHub:
- Replace `[Your Email]` placeholders
- Add your actual GitHub username to URLs
- Update contact information

### 3. Create GitHub Releases (Optional)

Tag your current state as v1.0:

```bash
git tag -a v1.0 -m "Initial release: LoRA and Frozen Encoder LOFO results"
git push origin v1.0
```

On GitHub:
1. Go to "Releases"
2. Click "Create a new release"
3. Select tag: v1.0
4. Release title: "v1.0 - LOFO Training Complete"
5. Description: Copy key results from README
6. Click "Publish release"

### 4. Add Topics/Tags

On GitHub repository page:
1. Click ⚙️ (Settings icon) next to "About"
2. Add topics:
   - `space-weather`
   - `solar-physics`
   - `transfer-learning`
   - `deep-learning`
   - `pytorch`
   - `foundation-models`
   - `lora`
   - `imf-prediction`

---

## 🔄 Making Updates Later

### To add more files or changes:

```bash
# Check what changed
git status

# Add specific files
git add filename.py

# Or add all changes
git add .

# Commit with message
git commit -m "Add new analysis script"

# Push to GitHub
git push
```

### To add more result CSVs later:

```bash
# Results CSVs are allowed through .gitignore
git add lofo_results/new_run/lofo_results.csv
git commit -m "Add new experiment results"
git push
```

---

## ⚠️ Important Notes

### What NOT to Push

The `.gitignore` is configured to prevent you from accidentally pushing:
- ❌ Large data files (data/sdo_files/, data/omni_data/*.csv)
- ❌ Model checkpoints (checkpoints/, *.pt, *.pth)
- ❌ LOFO fold results (lofo_results/*/fold_*/)
- ❌ Log files (*.log, nohup.out)

**If you try to push these, Git will ignore them automatically.**

### GitHub Size Limits

- **File size limit**: 100 MB per file
- **Repository size recommendation**: < 1 GB
- **Repository size limit**: 100 GB (soft limit)

Your current repository (~5-10 MB) is well within limits.

### If You Accidentally Add Large Files

```bash
# Remove file from Git but keep locally
git rm --cached filename

# Update .gitignore if needed
echo "filename" >> .gitignore

# Commit the removal
git commit -m "Remove large file from repository"
git push
```

---

## 📧 Sharing Your Repository

Once uploaded, share your work:

**Repository URL format**:
```
https://github.com/YOUR_USERNAME/imf-bz-forecasting-surya
```

**Include in**:
- Research presentations
- Paper acknowledgments
- CV/portfolio
- Academic applications

**Example share message**:
> "I've uploaded the complete IMF Bz forecasting project to GitHub:
> https://github.com/YOUR_USERNAME/imf-bz-forecasting-surya
>
> The repository includes all code, configurations, result summaries, and comprehensive documentation. Both LoRA and Frozen Encoder strategies achieved ~3.4 nT RMSE across 55 major flare events."

---

## 🎯 Quick Command Reference

```bash
# Check current status
git status

# See commit history
git log --oneline

# Check remote repository
git remote -v

# Pull latest changes (if working from multiple computers)
git pull

# Push your changes
git push

# Create new branch for experiments
git checkout -b experiment-name

# Switch back to main
git checkout main
```

---

## ✅ Verification Checklist

After pushing to GitHub, verify:

- [ ] README.md displays correctly on GitHub homepage
- [ ] All 22 files are present
- [ ] No large files (>10 MB) in repository
- [ ] Repository size shown is ~5-10 MB
- [ ] Commit message is clear and descriptive
- [ ] Code files (.py) are formatted correctly
- [ ] Configuration files (.yaml) are readable
- [ ] Result CSVs open correctly on GitHub

---

## 🤝 Making Repository Public vs Private

### Public Repository (Recommended for Research)
**Pros**:
- ✅ Demonstrates research productivity
- ✅ Enables collaboration
- ✅ Citable for papers
- ✅ Visible on your profile
- ✅ Can be featured in CV/portfolio

**Cons**:
- ⚠️ Anyone can see your code
- ⚠️ Need to be careful with sensitive data (already excluded via .gitignore)

### Private Repository
**Pros**:
- ✅ Code stays confidential
- ✅ Control who can access

**Cons**:
- ❌ Not discoverable by others
- ❌ Can't easily share (need to add collaborators)
- ❌ Doesn't show on public profile

**Recommendation**: Start **public** for research visibility and portfolio building. The code demonstrates good research practices and scientific rigor.

---

## 📞 Need Help?

**If you encounter errors**:

1. **Authentication failed**:
   - Use personal access token (see Option A above)
   - Or set up SSH key (see Option C)

2. **Remote already exists**:
   ```bash
   git remote remove origin
   # Then add again with correct URL
   ```

3. **Push rejected**:
   ```bash
   git pull origin main --rebase
   git push origin main
   ```

4. **Large file error**:
   - File is too large (>100 MB)
   - Add to .gitignore
   - Remove from git: `git rm --cached filename`

---

**You're ready to upload! Follow Step 1 and Step 2 above to get your research on GitHub. 🚀**

**Current Status**:
- ✅ Git initialized
- ✅ First commit created (22 files)
- ✅ Ready to push to GitHub
- ⏳ Waiting for GitHub repository creation
