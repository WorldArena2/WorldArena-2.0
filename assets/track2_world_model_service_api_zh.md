# WorldArena 2.0 Track 2：候选世界模型服务 API


## 1. 适用范围

本协议用于官方评测器查询参赛选手提交的动作条件世界模型。服务接收视觉历史和
未来动作序列，返回与动作逐步对齐的未来 RGB 帧。协议与具体模型架构、训练框架
和部署方式无关。

API v1 采用无状态设计：每次 `/v1/predict` 请求都携带完成预测所需的上下文，
服务不得依赖此前请求、请求顺序或服务端 session。无状态协议便于并发、重试、
隔离和复现实验。

## 2. 传输与认证

- 协议：HTTPS，HTTP/1.1 或 HTTP/2；
- 数据格式：UTF-8 JSON；
- 请求头：`Content-Type: application/json`；
- 认证：`Authorization: Bearer <token>`；
- API 基础地址：由参赛选手提交，例如 `https://wm.example.org`；
- 服务不得重定向到未登记域名，也不得在响应中返回外部下载 URL；
- token 通过赛事指定的安全渠道提交，不得写入仓库、镜像层或公开日志。

`GET /v1/health` 可以不要求认证，其余端点必须支持 Bearer token。认证失败返回
`401`，权限不足返回 `403`。

字符串使用 UTF-8。浮点输入必须是有限数，不接受 `NaN`、`Infinity` 或
`-Infinity`。

## 3. Endpoint

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/v1/health` | 存活与就绪检查 |
| `GET` | `/v1/capabilities` | 查询模型版本、profile 和服务限制 |
| `POST` | `/v1/predict` | 批量动作条件视频预测 |

评测器不依赖其他管理、调试或指标接口。

## 4. 通用约定

### 4.1 标识与版本

- `api_version`：固定为字符串 `"1.0"`；
- `model_version`：参赛选手定义的不可变模型版本；
- `profile_id`：官方发布的输入输出规格标识；
- `request_id`：评测器生成的 UUID；
- `sample_id`：请求内唯一的不透明字符串。

服务必须原样回传 `request_id`、`sample_id` 和 `model_version`，不得从不透明
标识推断任务或生成内容。

### 4.2 图像

API v1 的图像统一为 PNG 编码的 RGB `uint8`：

```json
{
  "encoding": "png_base64",
  "color_space": "RGB",
  "height": 256,
  "width": 256,
  "channels": 3,
  "data": "<BASE64_PNG_BYTES>"
}
```

约束：

- `data` 是 PNG 文件字节的标准 Base64，不含 `data:` 前缀和换行；
- 解码结果必须为 `height × width × 3`，通道顺序为 RGB；
- 禁止 alpha 通道、调色板图、灰度图、动画 PNG 和额外视频容器；
- 输入和输出尺寸必须符合当前 `profile_id`；
- EXIF、文本块等元数据不参与评测，服务不应写入此类信息。

### 4.3 时间与动作对齐

设：

- 历史帧：`context.frames = [o0, o1, ..., o(F-1)]`；
- 历史动作：`context.actions = [a0, ..., a(F-2)]`；
- 未来动作：`actions = [u0, ..., u(P-1)]`；
- 输出帧：`frames = [p0, ..., p(P-1)]`。

必须满足：

```text
len(context.actions) = len(context.frames) - 1
len(response.frames) = len(request.actions)

a_i: o_i -> o_(i+1)
u_0: o_(F-1) -> p_0
u_j: p_(j-1) -> p_j, j > 0
```

帧和动作均按时间从旧到新排列。服务不得返回条件帧、跳帧，或在首尾额外添加
重复帧。

动作向量使用 profile 定义的 canonical 表达，JSON shape 为
`[T, action_dim]`。服务不得自行改变关节顺序、单位、绝对/相对语义或再次
归一化；具体动作语义由官方 profile 说明。

### 4.4 可选状态与指令

`context.states` 若存在，shape 为 `[F, state_dim]`，每个状态与同索引历史帧
对齐。`instruction` 若存在则为自然语言字符串；省略和 `null` 等价。

仅当能力清单声明支持且官方 profile 要求时，评测器才会发送这些字段。API v1
不要求服务返回预测状态。

## 5. `GET /v1/health`

用于区分进程存活和模型就绪。成功时返回 `200`：

```json
{
  "status": "ready",
  "api_version": "1.0",
  "model_version": "wm-2026-07-01"
}
```

`status` 允许：

- `ready`：模型已加载，可以接收预测；
- `starting`：进程存活但暂不可预测；
- `degraded`：可以服务，但存在已知容量问题。

以上状态均可返回 `200`，但评测器只在 `ready` 时开始正式请求。未捕获故障返回
`5xx`。响应不得包含主机路径、硬件编号、内部地址、密钥或 traceback。

## 6. `GET /v1/capabilities`

成功时返回 `200`：

```json
{
  "api_version": "1.0",
  "model_version": "wm-2026-07-01",
  "profiles": [
    {
      "profile_id": "official-profile",
      "image": {
        "encoding": "png_base64",
        "color_space": "RGB",
        "height": 256,
        "width": 256,
        "channels": 3
      },
      "context_frames": {"min": 1, "max": 8},
      "prediction_frames": {"min": 1, "max": 16},
      "action": {
        "dtype": "float32",
        "dimension": 14,
        "representation": "profile_canonical"
      },
      "state": {"supported": false, "dimension": null},
      "instruction": {"supported": true}
    }
  ],
  "limits": {
    "max_batch_size": 8,
    "max_request_bytes": 33554432,
    "max_concurrency": 2,
    "recommended_timeout_ms": 600000
  },
  "determinism": {
    "seed_supported": true,
    "same_seed_same_pixels": true
  }
}
```

示例数值不构成正式 profile 承诺。参赛服务必须声明真实能力，并满足：

- `profiles` 至少包含一个官方要求的 `profile_id`；
- `context_frames.min >= 1`，`prediction_frames.min >= 1`；
- `action.dimension` 和可选 `state.dimension` 为正整数；
- `max_batch_size` 表示单次请求中 `samples` 的最大长度；
- `max_request_bytes` 包含完整 JSON 和 Base64 数据；
- `recommended_timeout_ms` 不能覆盖官方最大时限；
- 正式 profile 要求可复现时，两个 determinism 字段必须为 `true`。

评测开始后，模型版本和能力清单必须保持不变。

## 7. `POST /v1/predict`

### 7.1 请求

```json
{
  "api_version": "1.0",
  "request_id": "8da7108d-5f2f-4eb5-bf41-42a0db164011",
  "model_version": "wm-2026-07-01",
  "profile_id": "official-profile",
  "samples": [
    {
      "sample_id": "sample-0",
      "seed": 123456,
      "context": {
        "frames": [
          {
            "encoding": "png_base64",
            "color_space": "RGB",
            "height": 256,
            "width": 256,
            "channels": 3,
            "data": "<BASE64_PNG_BYTES>"
          }
        ],
        "actions": [],
        "states": null,
        "instruction": "perform the instructed manipulation"
      },
      "actions": [
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      ]
    }
  ]
}
```

字段约束：

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `api_version` | string | 是 | 必须为 `"1.0"` |
| `request_id` | string | 是 | UUID，用于关联、幂等和重试 |
| `model_version` | string | 是 | 必须与能力清单一致 |
| `profile_id` | string | 是 | 必须是已声明 profile |
| `samples` | array | 是 | 长度 `1..max_batch_size` |
| `samples[].sample_id` | string | 是 | 在本请求内唯一 |
| `samples[].seed` | integer | 是 | 0 到 2^63-1 |
| `context.frames` | image[] | 是 | 长度符合 profile |
| `context.actions` | float[][] | 是 | 长度为历史帧数减一 |
| `context.states` | float[][] 或 null | 否 | 与历史帧逐帧对齐 |
| `context.instruction` | string 或 null | 否 | 长度符合 profile |
| `samples[].actions` | float[][] | 是 | `[P, action_dim]` |

同一 batch 内所有样本使用同一个 profile，但历史长度和预测长度可以在 profile
范围内不同。服务必须先完整校验请求；任一样本不合法时，整个请求按第 9 节返回
错误，不得返回部分成功。

### 7.2 成功响应

成功时返回 `200`：

```json
{
  "api_version": "1.0",
  "request_id": "8da7108d-5f2f-4eb5-bf41-42a0db164011",
  "model_version": "wm-2026-07-01",
  "predictions": [
    {
      "sample_id": "sample-0",
      "frames": [
        {
          "encoding": "png_base64",
          "color_space": "RGB",
          "height": 256,
          "width": 256,
          "channels": 3,
          "data": "<BASE64_PNG_BYTES>"
        }
      ]
    }
  ]
}
```

约束：

- `predictions` 与请求 `samples` 一一对应，顺序必须相同；
- 每个 `sample_id` 必须原样回传且只能出现一次；
- 每个输出帧必须满足当前 profile 的图像规范；
- 每个样本的输出帧数必须等于对应未来动作数；
- 响应不得包含奖励、动作建议、成功判定、终止信号或外部 URL；
- 可增加 `diagnostics` 对象，但评测器会忽略它。

### 7.3 随机性与幂等

- `seed` 的作用域是单个 sample；
- 相同模型版本、profile、输入和 seed 必须生成相同的解码后 RGB 像素；
- batch 中其他样本的存在和顺序不得改变某个样本的结果；
- 同一 `request_id` 使用相同载荷重试时必须返回等价结果；
- 同一 `request_id` 携带不同载荷时返回 `409 REQUEST_ID_CONFLICT`；
- 服务重启前后不得因未声明的在线更新改变结果。

## 8. 并发、容量与超时

- 服务必须至少支持其声明的 `max_concurrency`；
- 超过容量时返回 `429 OVERLOADED` 和可选 `Retry-After`，不得无限排队；
- 服务应在 `recommended_timeout_ms` 内完成其声明的最大合法请求；
- 客户端断开后，服务应尽快取消无用计算；
- 评测器可以动态组 batch，也可以降级为单样本请求；
- 评测器会对 `429`、`502`、`503`、`504` 和网络中断进行有限重试；
- 参数、认证、版本和 profile 错误不会自动重试。

不得通过降低输出帧数、尺寸或精度规避容量限制。

## 9. 错误响应

所有非 2xx 响应使用统一 JSON：

```json
{
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "samples[0].actions has dimension 13; expected 14",
    "request_id": "8da7108d-5f2f-4eb5-bf41-42a0db164011",
    "retryable": false
  }
}
```

| HTTP | `code` | `retryable` | 场景 |
|---|---|---|---|
| 400 | `INVALID_ARGUMENT` | false | JSON、字段、shape、数值或图像非法 |
| 401 | `UNAUTHENTICATED` | false | token 缺失或无效 |
| 403 | `PERMISSION_DENIED` | false | token 无该服务权限 |
| 409 | `REQUEST_ID_CONFLICT` | false | 同一 request ID 对应不同载荷 |
| 413 | `PAYLOAD_TOO_LARGE` | false | 超过声明的请求大小 |
| 422 | `UNSUPPORTED_VERSION` | false | API、模型版本或 profile 不支持 |
| 429 | `OVERLOADED` | true | 并发或瞬时容量不足 |
| 500 | `INTERNAL` | true | 未预期推理错误 |
| 503 | `NOT_READY` | true | 模型尚未加载或暂不可用 |
| 504 | `DEADLINE_EXCEEDED` | true | 服务端截止时间已到 |

`message` 应足以定位字段问题，但不得暴露 traceback、文件路径、主机名、模型权重
路径、内部拓扑或凭据。无法解析 `request_id` 时可返回 `null`。

## 10. 版本兼容

- `1.x` 内新增字段必须是可选字段；
- 服务应忽略不认识的可选字段，但必须校验已知字段；
- 删除字段、改变语义或改变编码需要升级 major 版本；
- 收到不支持的 major 版本时返回 `422 UNSUPPORTED_VERSION`；
- 正式评测绑定 API、model 和 profile 三个版本，评测期间不得漂移。

## 11. 安全与隐私

- 只解析 JSON、Base64 和 PNG，禁止任意对象反序列化和动态代码执行；
- 对 JSON 深度、字符串长度、图片尺寸、解压后大小和 batch 大小设置上限；
- token 在日志中必须脱敏；
- 默认不记录请求图像、动作、状态、指令或完整响应；
- 禁止把请求载荷发送给未申报的第三方服务；
- 不得在输出图片或元数据中加入水印、追踪标识或隐写内容；
- 正式评测数据仅用于当次推理，不得用于训练、分析、共享或产品改进；
- 健康和错误响应不得泄露部署路径、硬件映射或内部网络信息。

## 12. 最小联调

健康检查：

```bash
curl --fail --silent --show-error \
  https://wm.example.org/v1/health
```

能力检查：

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${WORLD_MODEL_TOKEN}" \
  https://wm.example.org/v1/capabilities
```

预测请求：

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${WORLD_MODEL_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @predict_request.json \
  https://wm.example.org/v1/predict
```

提交前至少验证：

- [ ] 三个 endpoint 的状态码、认证和 JSON 类型正确；
- [ ] capabilities 与实际可接受范围一致；
- [ ] 单样本和最大 batch 均能完成；
- [ ] 历史帧/动作和预测帧/动作严格对齐；
- [ ] 非法 shape、损坏 PNG、错误版本返回规范错误；
- [ ] 相同输入与 seed 的像素结果可复现；
- [ ] 相同 request ID 重试幂等，不同载荷冲突；
- [ ] 并发超限明确返回 `429`；
- [ ] 日志、健康响应和错误响应不泄露敏感信息；
- [ ] 服务重启后模型版本和能力清单保持不变。

## 13. 提交信息

参赛选手通过指定渠道提交：

```text
base_url
bearer_token（通过单独的安全渠道提交）
api_version
model_version
required profile_id
容器镜像 digest（如由官方托管）
启动与资源需求
技术联系人
模型、数据和第三方依赖合规声明
```

请勿提交训练数据、内部路径、云密钥、SSH 凭据或与服务调用无关的运维权限。

