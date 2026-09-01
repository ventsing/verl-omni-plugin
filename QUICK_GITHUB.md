# 🚀 快速创建 GitHub 仓库

## 最简单的方式（3 步完成）

### 方式 1：使用自动化脚本（推荐）

```bash
cd /home/ventsing/source/opensource/ai/llm/verl-omni-plugin
./scripts/init_github.sh
```

脚本会自动：
- ✅ 检查 Git 配置
- ✅ 初始化 Git 仓库
- ✅ 添加所有文件
- ✅ 创建初始提交
- ✅ 引导你配置远程仓库
- ✅ 推送到 GitHub

### 方式 2：手动命令（一行搞定）

```bash
cd /home/ventsing/source/opensource/ai/llm/verl-omni-plugin && \
git init && \
git add . && \
git commit -m "Initial commit: verl-omni-plugin" && \
gh repo create verl-omni-plugin --public --source=. --push
```

**注意**: 需要安装 [GitHub CLI](https://cli.github.com/)

### 方式 3：完全手动

```bash
# 1. 进入目录
cd /home/ventsing/source/opensource/ai/llm/verl-omni-plugin

# 2. 初始化 Git
git init
git add .
git commit -m "Initial commit"

# 3. 在 GitHub 上创建仓库后
git remote add origin git@github.com:YOUR_USERNAME/verl-omni-plugin.git
git branch -M main
git push -u origin main
```

## 📋 前置准备

### 1. 配置 Git（首次使用）

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. 配置 SSH Key（推荐）

```bash
# 生成 SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# 添加到 ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 复制公钥
cat ~/.ssh/id_ed25519.pub
```

然后访问 https://github.com/settings/keys 添加 SSH key

### 3. 安装 GitHub CLI（可选但推荐）

```bash
# macOS
brew install gh

# Linux
sudo apt install gh

# 或使用官方安装脚本
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

## 📂 仓库内容预览

推送后，你的仓库将包含：

```
verl-omni-plugin/
├── 📄 README.md                      # 项目主文档
├── 📄 QUICKSTART.md                  # 快速开始指南
├── 📄 GITHUB_SETUP.md                # GitHub 设置指南
├── 📄 pyproject.toml                 # 项目配置
│
├── 📁 shared/                        # 共享工具
│   ├── patch_manager/                # Patch 管理器
│   ├── audio/                        # 音频处理
│   └── utils/                        # 通用工具
│
├── 📁 plugins/                       # 插件系统
│   ├── verl/                         # verl 插件
│   ├── verl_omni/                    # verl-omni 插件
│   ├── vllm/                         # vllm 插件
│   └── vllm_omni/                    # vllm-omni 插件
│
├── 📁 examples/                      # 示例代码
│   ├── audio_support_example.py      # 音频支持样例
│   ├── full_duplex_example.py        # 全双工样例
│   └── README.md
│
├── 📁 tests/                         # 测试代码
│   ├── test_audio.py
│   ├── test_patch_manager.py
│   └── test_reward.py
│
├── 📁 docs/                          # 架构文档
│   ├── README.md
│   ├── plugin_architecture_design.md
│   └── plugin_monkey_patch_analysis.md
│
└── 📁 scripts/                       # 工具脚本
    └── init_github.sh
```

## ✅ 检查清单

推送前请确认：

- [ ] Git 已安装并配置
- [ ] SSH key 已配置（或使用 HTTPS）
- [ ] 所有文件都已添加
- [ ] 初始提交已创建
- [ ] 远程仓库已配置
- [ ] 代码已推送到 GitHub

## 🔧 常用 Git 命令

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline --graph

# 添加和提交
git add .
git commit -m "Update message"

# 推送和拉取
git push origin main
git pull origin main

# 创建分支
git checkout -b feature-name

# 查看远程仓库
git remote -v
```

## 📚 详细文档

- [GITHUB_SETUP.md](./GITHUB_SETUP.md) - 完整的 GitHub 设置指南
- [README.md](./README.md) - 项目说明
- [QUICKSTART.md](./QUICKSTART.md) - 快速开始

## ❓ 遇到问题？

### 权限错误
```bash
# 检查 SSH key
ssh -T git@github.com

# 或使用 HTTPS
git remote set-url origin https://github.com/YOUR_USERNAME/verl-omni-plugin.git
```

### 推送失败
```bash
# 检查远程配置
git remote -v

# 重新设置远程
git remote remove origin
git remote add origin git@github.com:YOUR_USERNAME/verl-omni-plugin.git
```

### 分支问题
```bash
# 设置默认分支
git branch -M main

# 强制推送（谨慎使用）
git push -f origin main
```

## 🎉 完成！

推送成功后：
1. 访问你的 GitHub 仓库
2. 添加仓库描述和主题标签
3. 设置仓库可见性（Public/Private）
4. 分享给其他人！

---

**提示**: 推荐使用方式 1（自动化脚本）或方式 2（GitHub CLI），最简单快捷！
