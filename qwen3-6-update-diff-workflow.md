# qwen3.6 更新 diff 流程

本文记录 `vllm-ascend` 仓库中 `qwen3.6` 分支基于 `tallmessiwu/main` 更新 diff 的常用操作。

仓库路径：

```bash
cd /home/omni/code/vllm-vllm-ascend/vllm-ascend
```

## 1. 确认当前状态

先确认当前分支、远端和工作区是否干净：

```bash
git status --short --branch
git remote -v
git branch -vv
```

预期当前在：

```text
qwen3.6
```

如果工作区有未提交改动，先处理掉再继续，避免 `reset --hard` 或 `rebase` 覆盖本地文件。

## 2. 拉取 tallmessiwu/main

```bash
git fetch tallmessiwu main
```

确认目标提交：

```bash
git rev-parse --short tallmessiwu/main
git log --oneline --decorate --max-count=5 tallmessiwu/main
```

## 3. 查看 qwen3.6 相对 tallmessiwu/main 的差异

查看当前 `qwen3.6` 比 `tallmessiwu/main` 多了哪些提交：

```bash
git log --oneline tallmessiwu/main..qwen3.6
```

查看 diff 文件统计：

```bash
git diff --stat tallmessiwu/main...qwen3.6
git diff --name-status tallmessiwu/main...qwen3.6
```

## 4. 场景 A：强制用 tallmessiwu/main 覆盖 qwen3.6

如果目标是让 `qwen3.6` 完全等于 `tallmessiwu/main`，直接执行：

```bash
git checkout qwen3.6
git reset --hard tallmessiwu/main
```

验证：

```bash
git rev-parse --short HEAD
git rev-parse --short tallmessiwu/main
git diff --stat tallmessiwu/main..qwen3.6
git status --short --branch
```

如果 `git diff --stat tallmessiwu/main..qwen3.6` 没有输出，说明本地 `qwen3.6` 已经和 `tallmessiwu/main` 一致。

如果还需要覆盖远端 `origin/qwen3.6`，执行：

```bash
git push --force-with-lease origin qwen3.6
```

注意：`reset --hard` 会丢弃本地 `qwen3.6` 上独有的提交；`push --force-with-lease` 会改写远端分支历史。

## 5. 场景 B：保留 qwen3.6 独有提交，只更新 base

如果目标是保留 `qwen3.6` 自己的提交，并把它们移动到最新 `tallmessiwu/main` 之上，用 rebase。

先找到旧分叉点：

```bash
git merge-base qwen3.6 tallmessiwu/main
```

假设旧分叉点是 `<old-base>`，执行：

```bash
git checkout qwen3.6
git rebase --onto tallmessiwu/main <old-base> qwen3.6
```

如果有冲突：

```bash
git status
```

手动解决冲突后继续：

```bash
git add <resolved-files>
git rebase --continue
```

如果判断 rebase 方向错了，可以中止：

```bash
git rebase --abort
```

验证：

```bash
git log --oneline tallmessiwu/main..qwen3.6
git diff --stat tallmessiwu/main...qwen3.6
```

确认无误后推送远端：

```bash
git push --force-with-lease origin qwen3.6
```

## 6. 建议的安全备份

在执行 `reset --hard` 或 rebase 前，可以先创建备份分支：

```bash
git branch qwen3.6-before-update qwen3.6
```

如果后续需要恢复：

```bash
git checkout qwen3.6
git reset --hard qwen3.6-before-update
```

## 7. 本次实际执行过的核心操作

本次需求是“强制用 `tallmessiwu/main` 覆盖 `qwen3.6`”，实际流程为：

```bash
cd /home/omni/code/vllm-vllm-ascend/vllm-ascend
git fetch tallmessiwu main
git checkout qwen3.6
git reset --hard tallmessiwu/main
git diff --stat tallmessiwu/main..qwen3.6
```

最终本地 `qwen3.6` 与 `tallmessiwu/main` 一致。若要同步 GitHub 上的 `origin/qwen3.6`，还需要执行：

```bash
git push --force-with-lease origin qwen3.6
```
