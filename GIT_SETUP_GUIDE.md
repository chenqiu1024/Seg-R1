# Git Push 配置指南

## 问题诊断

您的 Git 配置：
- ✅ 用户名和邮箱已配置
- ✅ 凭证助手已设置为 `store`
- ✅ 远程仓库已配置
- ❌ 凭证缺失或过期

## 解决方案

### 方法 1: 使用 GitHub Personal Access Token（推荐）

#### 步骤 1: 生成 GitHub Personal Access Token

1. 访问 GitHub: https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置：
   - **Note**: `Seg-R0 Development`
   - **Expiration**: 90 days（或根据需要）
   - **Scopes**: 勾选 `repo`（完整仓库访问权限）
4. 点击 "Generate token"
5. **复制生成的 token**（只显示一次，请保存好）

#### 步骤 2: 配置 Git 使用 Token

**选项 A: 通过 push 时输入（推荐）**

```bash
# 直接 push，会提示输入凭证
git push mine peft-v1

# 提示时输入：
# Username: chenqiu1024
# Password: <粘贴您的 Personal Access Token>

# 凭证会自动保存，下次不需要再输入
```

**选项 B: 在 URL 中包含 token**

```bash
# 设置远程 URL 包含 token（不推荐，token 可见）
git remote set-url mine https://<YOUR_TOKEN>@github.com/chenqiu1024/Seg-R1.git

# 然后 push
git push mine peft-v1
```

**选项 C: 使用 gh CLI（如果已安装）**

```bash
# GitHub CLI 认证
gh auth login

# 然后正常 push
git push mine peft-v1
```

### 方法 2: 切换到 SSH（更安全，长期推荐）

#### 步骤 1: 生成 SSH Key

```bash
# 生成新的 SSH key
ssh-keygen -t ed25519 -C "donchiu@redshore.space"

# 按提示操作：
# - 文件位置：直接回车（使用默认 ~/.ssh/id_ed25519）
# - 密码：可选（建议设置）

# 查看公钥
cat ~/.ssh/id_ed25519.pub
```

#### 步骤 2: 添加 SSH Key 到 GitHub

1. 复制上面命令输出的公钥（以 `ssh-ed25519` 开头）
2. 访问 GitHub: https://github.com/settings/ssh/new
3. Title: `Seg-R0 Server`
4. Key: 粘贴公钥
5. 点击 "Add SSH key"

#### 步骤 3: 测试 SSH 连接

```bash
ssh -T git@github.com

# 应该看到：
# Hi chenqiu1024! You've successfully authenticated...
```

#### 步骤 4: 切换远程 URL 到 SSH

```bash
# 切换到 SSH URL
git remote set-url mine git@github.com:chenqiu1024/Seg-R1.git

# 验证
git remote -v

# Push
git push mine peft-v1
```

## 当前状态

- **当前分支**: peft-v1
- **远程仓库**: mine (https://github.com/chenqiu1024/Seg-R1.git)
- **待推送**: 3 个 commits ahead

## 推荐操作

### 快速方案（方法 1，选项 A）

```bash
# 1. 生成 GitHub PAT（见上面步骤）
# 2. 直接 push
git push mine peft-v1

# 3. 输入凭证：
#    Username: chenqiu1024
#    Password: <您的 Personal Access Token>

# 4. 完成！凭证会自动保存
```

### 长期方案（方法 2）

配置 SSH key，以后不需要每次输入 token。

## 其他有用命令

### 查看待推送的内容

```bash
# 查看待推送的 commits
git log mine/peft-v1..HEAD --oneline

# 查看改动
git diff mine/peft-v1..HEAD --stat
```

### 如果想 push 到不同分支

```bash
# Push 到远程的同名分支
git push mine peft-v1

# Push 到远程的不同分支
git push mine peft-v1:my-feature-branch

# Push 所有分支
git push mine --all
```

### 凭证问题排查

```bash
# 查看存储的凭证
cat ~/.git-credentials

# 清除凭证（如果需要重新配置）
rm ~/.git-credentials

# 测试凭证
git ls-remote mine
```

## 安全提示

1. **Personal Access Token** 就像密码，不要分享
2. **设置合适的过期时间**（建议 90 天）
3. **定期更新 token**
4. **使用 SSH 更安全**（长期推荐）

## 下一步

我已经帮您退出了 bisect 模式并切回 peft-v1 分支。

现在您需要：
1. 生成 GitHub Personal Access Token（见上面步骤）
2. 运行 `git push mine peft-v1`
3. 输入 token 作为密码

需要我执行 push 命令吗？（您需要先准备好 token）

