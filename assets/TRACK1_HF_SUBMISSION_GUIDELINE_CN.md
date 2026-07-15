# WorldArena 2.0 Track 1 Hugging Face 提交规范


本文档面向参赛者，说明 Track 1 视频质量评测的提交格式、文件组织、`model_readme.md` 元信息要求，以及如何将结果上传到 Hugging Face dataset repo 供组织方自动评测。

组织方评测系统会自动从你提供的公开 Hugging Face dataset repo 下载唯一的 `.tar.gz` 或 `.tgz` 提交包，并完成解压、校验、评测和结果归档。因此，请严格按照本文档准备提交内容。

---

## 目录

- [1. 任务目标](#1-任务目标)
- [2. 视频生成要求](#2-视频生成要求)
- [3. Hugging Face 仓库要求](#3-hugging-face-仓库要求)
- [4. 提交压缩包格式](#4-提交压缩包格式)
- [5. model_readme.md 规范](#5-model_readmemd-规范)
- [6. 打包与上传示例](#6-打包与上传示例)
- [7. 提交前自检清单](#7-提交前自检清单)
- [8. 常见无效提交](#8-常见无效提交)
- [9. 评测结果说明](#9-评测结果说明)

---

## 1. 任务目标

Track 1 评测世界模型生成视频的质量。参赛者需要基于 WorldArena 2.0 官方测试集（https://huggingface.co/datasets/WorldArena/WorldArena2.0/blob/main/dataset_track1.tar.gz），为每个 episode 生成对应 `.mp4` 视频，并将所有视频和模型说明文件打包提交。

组织方固定使用以下测试集进行评测：

```text
https://huggingface.co/datasets/WorldArena/WorldArena2.0/blob/main/dataset_track1.tar.gz
```

参赛者只需要提交生成视频和模型元信息。

---

## 2. 视频生成要求

| 项目 | 要求 |
|------|------|
| 视频格式 | `.mp4` |
| 数据划分 | WorldArena Track 1 官方 test split |
| 分辨率 | 推荐 `640x480` 或更高 |
| 帧率 | 推荐 `24` fps |
| Text-driven 长度 | 建议与对应 GT trajectory 长度对齐 |
| Action-driven 长度 | 建议与对应 GT trajectory 长度对齐 |
| 文件位置 | 所有视频必须放在压缩包内的 `videos/` 目录下 |

建议视频文件名清晰可追踪，例如：

```text
videos/
├── episode_000001.mp4
├── episode_000002.mp4
├── episode_000003.mp4
└── ...
```

> 注意：请不要在 `videos/` 目录外放置用于评测的视频。

---

## 3. Hugging Face 仓库要求

参赛者必须创建一个 Hugging Face **dataset** repo，并上传一个提交压缩包。

| 项目 | 要求 |
|------|------|
| Repo 类型 | 必须是 `dataset` |
| Repo 权限 | 必须公开可下载 |
| 授权方式 | 不支持 token，不支持私有 repo，不支持 gated repo |
| 压缩包数量 | Repo 根目录必须且只能有一个 `.tar.gz` 或 `.tgz` |
| 压缩包位置 | 必须位于 repo 根目录，不能放在子目录 |

有效示例：

```text
your-hf-dataset-repo/
└── submission.tar.gz
```

无效示例：

```text
your-hf-dataset-repo/
├── submission_v1.tar.gz
└── submission_v2.tar.gz
```

```text
your-hf-dataset-repo/
└── release/
    └── submission.tar.gz
```

---

## 4. 提交压缩包格式

提交包必须是 `.tar.gz` 或 `.tgz` 格式，并包含：

1. `model_readme.md`
2. `videos/`

推荐结构：

```text
submission.tar.gz
├── model_readme.md
└── videos/
    ├── episode_000001.mp4
    ├── episode_000002.mp4
    └── ...
```

也允许压缩包内多一层父目录，但该父目录下仍必须包含 `model_readme.md` 和 `videos/`：

```text
submission.tar.gz
└── MyModel_eval/
    ├── model_readme.md
    └── videos/
        ├── episode_000001.mp4
        └── ...
```

为了避免解压安全检查失败，请确保压缩包内不要包含：

- 绝对路径，例如 `/home/user/file.mp4`
- 上级目录跳转路径，例如 `../file.mp4`
- 软链接或硬链接
- 设备文件、FIFO 等特殊文件
- 多个 `model_readme.md`

---

## 5. `model_readme.md` 规范

`model_readme.md` 必须以 YAML metadata block 开头。自动评测系统会从这里读取模型名称、版本、组织、年份、开源状态和驱动类型。

### 5.1 必填模板

```markdown
---
model_name: example_model
version: v1.2
organization: Example Lab
release_year: 2026
source_type: open_source
control_type: text_driven
license: Apache-2.0
paper_url: https://example.com/paper
code_url: https://github.com/example/example_model
---

# example_model v1.2

example_model v1.2 is a text-driven world model for robotic video generation.

## Submission Notes

The submitted videos correspond to the official WorldArena Track 1 test split.
```

### 5.2 字段要求

| 字段 | 是否必填 | 合法值或说明 |
|------|----------|--------------|
| `model_name` | 必填 | 模型名称，例如 `example_model` |
| `version` | 必填 | 模型版本，例如 `v1.2`、`2026-06-ckpt` |
| `organization` | 必填 | 团队、学校、公司或实验室名称 |
| `release_year` | 必填 | 四位年份，例如 `2026` |
| `source_type` | 必填 | 只能是 `open_source` 或 `closed_source` |
| `control_type` | 必填 | 只能是 `text_driven`、`action_driven` 或 `hybrid` |
| `license` | 可选 | 例如 `Apache-2.0`、`MIT`、`Proprietary` |
| `paper_url` | 可选 | 论文链接 |
| `code_url` | 可选 | 代码仓库链接 |

### 5.3 字段含义

`source_type` 表示模型是否开源：

| 值 | 含义 |
|----|------|
| `open_source` | 模型代码、权重或核心实现公开 |
| `closed_source` | 模型代码、权重或核心实现未公开 |

`control_type` 表示模型生成视频时的驱动方式：

| 值 | 含义 |
|----|------|
| `text_driven` | 主要由文本指令驱动 |
| `action_driven` | 主要由动作序列驱动 |
| `hybrid` | 同时使用文本和动作等多种条件 |

### 5.4 示例文件

组织方提供了一个示例：

```text
https://huggingface.co/datasets/WorldArena/WorldArena2.0/blob/main/examplev1.tar.gz
```

---

## 6. 打包与上传示例

### 6.1 本地目录准备

建议先整理成本地目录：

```text
example_model_eval/
├── model_readme.md
└── videos/
    ├── episode_000001.mp4
    ├── episode_000002.mp4
    └── ...
```

### 6.2 生成 `.tar.gz`

```bash
tar -czf submission.tar.gz -C example_model_eval .
```

检查压缩包内容：

```bash
tar -tzf submission.tar.gz | head
```

你应该能看到类似：

```text
./model_readme.md
./videos/
./videos/episode_000001.mp4
./videos/episode_000002.mp4
```

### 6.3 上传到 Hugging Face dataset repo

方式一：使用网页上传。

1. 在 Hugging Face 创建一个 dataset repo。
2. 确认 repo 是 public。
3. 上传 `submission.tar.gz` 到 repo 根目录。
4. 确认 repo 根目录只有这一个 `.tar.gz` 或 `.tgz` 提交包。

方式二：使用命令行上传。

```bash
huggingface-cli upload <user-or-org>/<dataset-repo> \
  submission.tar.gz \
  submission.tar.gz \
  --repo-type dataset
```

上传后，请将 Hugging Face dataset repo id 和 revision 提交给组织方，例如：

```text
hf_repo_id: your-org/example-model-track1-submission
hf_revision: main
```

---

## 7. 提交前自检清单

提交前请逐项确认：

- Hugging Face repo 类型是 `dataset`。
- Hugging Face repo 是 public，不需要 token，不是 gated repo。
- Repo 根目录只有一个 `.tar.gz` 或 `.tgz` 文件。
- 压缩包内包含 `model_readme.md`。
- 压缩包内包含 `videos/`。
- `videos/` 下至少有一个 `.mp4`。
- `model_readme.md` 位于压缩包内，并且以 YAML metadata block 开头。
- `model_name`、`version`、`organization`、`release_year`、`source_type`、`control_type` 都已填写。
- `release_year` 是四位年份。
- `source_type` 是 `open_source` 或 `closed_source`。
- `control_type` 是 `text_driven`、`action_driven` 或 `hybrid`。
- 压缩包内没有绝对路径、`..` 路径、链接文件或特殊文件。

本地快速检查命令：

```bash
tar -tzf submission.tar.gz | grep -E '(^|/)model_readme\.md$'
tar -tzf submission.tar.gz | grep -E '(^|/)videos/'
tar -tzf submission.tar.gz | grep -E '\.mp4$' | head
```

---

## 8. 常见无效提交

以下情况会导致自动评测失败：

| 失败原因 | 说明 |
|----------|------|
| `hf_repo_not_public_or_not_downloadable` | repo 私有、gated、缺失、未授权，或无法无 token 下载 |
| `missing_tar_gz` | repo 根目录没有 `.tar.gz` 或 `.tgz` |
| `multiple_tar_gz_found` | repo 根目录存在多个 `.tar.gz` 或 `.tgz` |
| `unsafe_tar_path` | 压缩包内包含绝对路径、`..`、链接或特殊文件 |
| `extract_failed` | 压缩包损坏或无法解压 |
| `missing_model_readme` | 缺少 `model_readme.md` |
| `metadata_invalid` | metadata block 缺失、字段缺失或字段值不合法 |
| `missing_videos` | 缺少 `videos/` 目录 |
| `empty_video_dir` | `videos/` 目录下没有 `.mp4` |
| `evaluation_failed` | 提交包通过校验，但官方评测过程失败 |

---

> 提醒：每一个同名模型最多只能提交三次，超出三次的提交将视作无效提交，高峰期重复提交可能会降低评测优先级，请尽量在提交前完成本地自检。
