# GitHub 仓库创建和推送指南

本指南帮助你创建 GitHub 仓库并推送 verl-omni-plugin 代码。

## 📋 前置条件

- [ ] GitHub 账号
- [ ] Git 已安装并配置
- [ ] SSH key 已配置（推荐）或 HTTPS 凭证

## 🚀 快速开始（推荐）

### 方式 1：使用 GitHub CLI（最简单）

```bash
# 1. 进入项目目录
cd /home/ventsing/source/opensource/ai/llm/verl-omni-plugin

# 2. 初始化 Git 仓库（如果还没有）
git init
git add .
git commit -m "Initial commit: verl-omni-plugin framework"

# 3. 创建 GitHub 仓库并推送
gh repo create verl-omni-plugin --public --source=. --push
```

### 方式 2：手动创建（标准方式）

#### 步骤 1：在 GitHub 上创建仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `verl-omni-plugin`（或其他你喜欢的名字）
   - **Description**: `Plugin framework for verl, verl-omni, vllm, and vllm-omni with audio and full-duplex support`
   - **Public** 或 **Private**（根据需要选择）
   - **不要**勾选 "Initialize this repository with a README"
3. 点击 "Create repository"

#### 步骤 2：初始化本地仓库并推送

```bash
# 1. 进入项目目录
cd /home/ventsing/source/opensource/ai/llm/verl-omni-plugin

# 2. 初始化 Git 仓库
git init

# 3. 添加所有文件
git add .

# 4. 创建初始提交
git commit -m "Initial commit: verl-omni-plugin framework

Features:
- Audio processing and model support
- Full-duplex training and inference
- Multimodal reward management
- Plugin system for verl, verl-omni, vllm, vllm-omni
- Comprehensive examples and tests

See README.md for details."

# 5. 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin git@github.com:YOUR_USERNAME/verl-omni-plugin.git
# 或使用 HTTPS
# git remote add origin https://github.com/YOUR_USERNAME/verl-omni-plugin.git

# 6. 推送到 GitHub
git branch -M main
git push -u origin main
```

## 📝 详细步骤

### 1. 配置 Git（如果还没配置）

```bash
# 设置用户名和邮箱
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置默认分支
git config --global init.defaultBranch main
```

### 2. 配置 SSH Key（推荐）

```bash
# 生成 SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# 添加到 ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 复制公钥到 GitHub
cat ~/.ssh/id_ed25519.pub
# 然后访问 https://github.com/settings/keys 添加 SSH key
```

### 3. 初始化仓库

```bash
cd /home/ventsing/source/opensource/ai/llm/verl-omni-plugin

# 初始化 Git
git init

# 添加 .gitignore（已存在）
# 添加所有文件
git add .

# 查看状态
git status

# 创建提交
git commit -m "Initial commit"
```

### 4. 创建 GitHub 仓库

访问 https://github.com/new 并创建仓库，然后：

```bash
# 添加远程仓库
git remote add origin git@github.com:YOUR_USERNAME/verl-omni-plugin.git

# 推送
git branch -M main
git push -u origin main
```

## 🔧 常用 Git 命令

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 添加文件
git add <file>
git add .  # 添加所有

# 提交
git commit -m "message"

# 推送到远程
git push origin main

# 拉取远程更新
git pull origin main

# 查看远程仓库
git remote -v

# 创建新分支
git checkout -b feature-name

# 切换分支
git checkout branch-name
```

## 📊 仓库结构预览

推送后，你的仓库将包含：

```
verl-omni-plugin/
├── README.md                    # 项目主文档
├── QUICKSTART.md                # 快速开始指南
├── pyproject.toml               # 项目配置
├── .gitignore                   # Git 忽略配置
│
├── shared/                      # 共享工具
│   ├── patch_manager/           # Patch 管理器
│   ├── audio/                   # 音频处理
│   └── utils/                   # 通用工具
│
├── plugins/                     # 插件
│   ├── verl/                    # verl 插件
│   ├── verl_omni/               # verl-omni 插件
│   ├── vllm/                    # vllm 插件
│   └── vllm_omni/               # vllm-omni 插件
│
├── examples/                    # 示例代码
│   ├── audio_support_example.py
│   ├── full_duplex_example.py
│   └── README.md
│
├── tests/                       # 测试
│   ├── test_audio.py
│   ├── test_patch_manager.py
│   └── test_reward.py
│
└── docs/                        # 文档
    ├── README.md
    ├── plugin_architecture_design.md
    └── plugin_monkey_patch_analysis.md
```

## 🎯 后续操作

### 添加更多功能

```bash
# 修改代码后
git add .
git commit -m "Add new feature"
git push origin main
```

### 创建发布版本

```bash
# 创建标签
git tag -a v0.1.0 -m "Version 0.1.0: Initial release"

# 推送标签
git push origin v0.1.0
```

### 贡献指南

创建 `CONTRIBUTING.md` 文件说明如何贡献代码。

## ❓ 常见问题

### Q: 推送时遇到权限错误？
**A**: 检查 SSH key 配置或使用 HTTPS + Personal Access Token

### Q: 如何更新已推送的代码？
**A**: 
```bash
git add .
git commit -m "Update message"
git push origin main
```

### Q: 如何删除远程仓库？
**A**: 在 GitHub 仓库设置页面删除

### Q: 如何设置仓库为私有？
**A**: 在 GitHub 仓库设置中修改可见性

## 🔗 相关资源

- [GitHub 文档](https://docs.github.com/)
- [Git 官方文档](https://git-scm.com/doc)
- [GitHub CLI 文档](https://cli.github.com/manual/)

## 📞 需要帮助？

如果遇到问题：
1. 检查 Git 配置是否正确
2. 确认 SSH key 或 HTTPS 凭证
3. 查看 GitHub 仓库设置
4. 参考 GitHub 文档

---

**提示**: 推荐使用 GitHub CLI (`gh`) 简化操作！
