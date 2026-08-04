# NLM 轨迹评分与强化学习 Reward 机制设计

## 1. 文档目的

本文档描述如何复用 NLM-CutPlan 的状态管理、动作应用、目标检测和启发式计算能力，为基于 veRL/Ray 的 GRPO、DAPO 等强化学习流程提供可验证的轨迹评分信号。

本文档是后续代码实现的接口与行为规格，重点回答以下问题：

1. 训练过程中如何启动、复用、回收和终止 NLM 状态评分进程。
2. 如何从一条 LLM 回复中提取动作，并复用现有动作回插逻辑逐步应用动作。
3. C++ 状态评分器应向 Python 返回哪些逐状态信息。
4. 如何根据合法前缀、目标到达、重复状态和启发式变化计算 reward。
5. 如何保证单样本即时评分、样本隔离、错误可恢复和结果可复现。

本文档不规定最终 reward 权重。权重应通过实验配置控制，而不应硬编码在 C++ 中。

## 2. 使用场景与约束

训练阶段的基本流程是：

```text
vLLM 生成一条 completion
        |
        v
veRL RewardLoop 立即提交该样本
        |
        v
Python 安全解析动作函数块
        |
        v
NLM 对动作序列逐步模拟和评分
        |
        v
Python 组合 scalar reward
        |
        v
veRL / DAPO 后续更新
```

需要遵守以下约束：

- 每条 completion 到达后立即评分，不等待同 prompt 的其他 completion，不额外引入 group barrier。
- NLM 评分只负责验证和产生原始信号，不负责组内归一化、group dropout 或 DAPO 的 advantage 计算。
- 不为每条 completion 重新执行 PDDL translator 和 preprocessor。
- C++ 返回可解释的原始评分数据；Python 负责 reward 公式和权重。
- 同一条 completion 内的状态新颖性必须独立计算，不得受其他样本的先后顺序影响。
- 评分流程不得修改搜索器 Open List，也不得产生搜索扩展副作用。

## 3. 总体架构

推荐增加四个逻辑层：

```text
NLMRewardManager (veRL)
    -> ActionParser
    -> NLMScorerPool (Ray Actors)
    -> nlm-score (C++ process)
```

### 3.1 NLMRewardManager

veRL 侧自定义 `RewardManagerBase`，负责接收单条 `DataProto`，提取：

- `solution_str`：模型原始回复。
- `problem_id`：问题唯一编号。
- `task_key`：预处理任务缓存键。
- `task_sas_path`：对应的 SAS 任务文件。
- `reward_config_version`：本次训练使用的 reward 配置版本。

`run_single()` 应设计为异步方法。它不等待其他样本，只把当前请求提交给 scorer pool，并在结果返回后计算 scalar reward。

### 3.2 NLMScorerPool

由多个 CPU Ray Actor 组成。请求根据 `problem_id` 或 `task_key` 做一致性路由：

```text
shard = hash(task_key) % actor_count
```

这样同一问题的请求倾向于到达同一个 Actor，可以复用已加载的任务上下文，但不需要等待组成 batch。

每个 Actor 维护一个有界 LRU：

```text
task_key -> nlm-score 子进程
```

由于当前 NLM 使用大量任务级全局变量，一个 C++ 子进程只能安全地绑定一个 planning task。不得在同一 NLM 进程中动态切换多个问题。

### 3.3 nlm-score

新增不带 Open List 的 C++ 轨迹评分入口，建议支持两种模式：

```bash
nlm-score --task problem.sas --once
nlm-score --task problem.sas --stream
```

- `--once`：读取一个请求，返回结果后退出，适合测试和性能基线。
- `--stream`：通过 JSONL 连续处理请求，适合 Ray Actor 缓存。

两种模式必须使用同一协议和同一核心动作模拟函数，确保切换生命周期不会改变评分语义。

## 4. 任务预处理与缓存

训练前应把每个 domain/problem 转换为 SAS，并缓存结果。缓存键至少包含：

```text
SHA256(domain PDDL
     + problem PDDL
     + translator/preprocessor version
     + relevant translation options)
```

建议目录布局：

```text
reward_task_cache/
  <task_hash>/
    task.sas
    metadata.json
```

`metadata.json` 至少记录：

- `problem_id`
- `task_hash`
- domain/problem 原始路径或内容哈希
- NLM commit
- translator/preprocessor commit
- 生成时间

RewardLoop 中不得重复翻译 PDDL。评分器应直接加载 `task.sas`。

## 5. 评分进程的启动、复用与终止

### 5.1 训练启动

1. 主训练进程启动 veRL 和 RewardLoop workers。
2. 初始化固定数量的 `NLMScorerActor`，每个 Actor 声明明确的 CPU 配额，不占用 GPU。
3. Actor 启动时不必加载所有问题，只初始化空的 LRU context cache。
4. 第一个请求到达某个 `task_key` 时，Actor 启动对应的 `nlm-score --stream --task ...` 子进程。
5. Actor 等待子进程返回 `ready` 握手后再发送评分请求。

建议握手信息包含：

```json
{
  "type": "ready",
  "protocol_version": 1,
  "task_hash": "...",
  "heuristic_config": "lmcutnumeric(...)"
}
```

### 5.2 请求期间

- 单个 `nlm-score` 默认串行处理请求，因为当前状态注册器、启发式缓存和全局变量不保证线程安全。
- Ray Actor 可以同时管理多个不同 task 的子进程，使不同问题并发评分。
- 同一 task 成为热点且队列过长时，可以为其启动多个独立副本；副本之间不得共享 `StateID`。
- 请求必须携带 `request_id`，响应必须原样返回，避免超时重试后结果错配。

### 5.3 正常终止

训练正常结束时：

1. Reward manager 停止接收新请求。
2. 等待已提交请求完成，或达到 shutdown grace period。
3. Actor 向所有 stream 子进程发送 `shutdown` 消息并关闭 stdin。
4. 等待子进程正常退出。
5. 超时后先发送 `SIGTERM`，再次超时才发送 `SIGKILL`。

### 5.4 有界回收

任一条件达到时应回收并按需重启子进程：

- 空闲时间超过 `idle_ttl_seconds`。
- 已处理请求数超过 `max_requests_per_process`。
- StateRegistry 状态数超过 `max_registered_states`。
- 进程 RSS 超过 `max_rss_mb`。
- 连续协议错误或内部错误超过阈值。
- task cache 被更新，`task_hash` 不再匹配。

回收阈值必须记录到训练配置和日志中。

### 5.5 异常与重试

- 子进程启动失败：返回 `scorer_start_error`，不生成伪造的部分 reward。
- 子进程在请求中退出：Actor 可以重启一次并重放请求。
- 重放仍失败：返回 `scorer_crash`。
- 请求超时：终止并重启该子进程，返回 `scorer_timeout`。
- 协议输出不是合法 JSON：保存 stderr 和原始输出，返回 `protocol_error`。

重试必须保持幂等。每条请求从原始初始状态重新模拟，不能从失败进程中的中间状态继续。

## 6. 从 LLM 回复到动作序列

### 6.1 Python 侧解析

复用当前 `hybrid_planner/validation/response_processor.py` 中的安全 AST 解析思路：

1. 从模型回复中定位 Python 代码或动作函数块。
2. 使用 `ast.parse()` 解析，不执行模型生成代码。
3. 仅接受顶层 `action_Xxx(...)` 调用。
4. 参数仅接受约定的对象标识符或字符串形式。
5. 保留模型输出顺序。
6. 使用 `_python_action_to_pddl()` 转换为 grounded PDDL action。

解析结果示例：

```text
action_Lift(hoist0, crate1, crate3, depot0)
action_Load(hoist0, crate1, truck0, depot0)
```

转换为：

```text
(lift hoist0 crate1 crate3 depot0)
(load hoist0 crate1 truck0 depot0)
```

如果模型回复无法解析，Python 不应调用 C++，直接产生：

```text
parse_ok = false
generated_action_count = 0
stop_reason = parse_error
```

### 6.2 是否保留 Unified Planning 验证

强化学习评分路径中，C++ NLM 应作为动作合法性、状态转移和目标判断的权威来源。Unified Planning 可以保留为离线一致性测试，但不建议在每个 reward 请求中先验证一次再由 C++ 重复验证。

这可以减少 CPU 开销，并避免 UP 与 NLM 数值语义存在细微差异时产生双重标准。

## 7. 复用现有动作回插逻辑

当前搜索器中的 `EagerSearch::inject_llm_action_chain()` 已实现主要动作模拟逻辑：

1. 从源 `StateID` 取出 `GlobalState`。
2. 标准化动作字符串并在 grounded operator map 中查找动作。
3. 调用 `GlobalOperator::is_applicable()` 判断动作是否可用。
4. 调用 `StateRegistry::get_successor_state()` 生成并注册后继状态。
5. 创建 `EvaluationContext` 并计算启发式值。
6. 更新 SearchNode、Open List 和搜索统计。

后续实现应从该函数中抽取无搜索副作用的公共组件，例如：

```text
ActionChainEvaluator::evaluate(initial_state, actions, options)
```

搜索回插和 reward scorer 都调用该组件：

```text
ActionChainEvaluator
    -> 搜索模式：把合法后继交给 SearchNode/Open List
    -> 评分模式：只收集状态信息并返回
```

评分模式必须禁止以下行为：

- 不创建或修改 SearchNode。
- 不插入 Open List。
- 不 reopen closed node。
- 不修改 pending/suspended 集合。
- 不更新 frontier trigger。
- 不改变搜索统计中的 expanded/generated 语义。

### 7.1 每一步动作的处理顺序

评分器从 `g_initial_state()` 开始，并先记录状态序号 0。随后对第 `i` 个动作执行：

1. 标准化动作名称。
2. 查找 grounded operator；找不到则以 `unknown_action` 截断。
3. 检查适用性；不适用则以 `inapplicable_action` 截断。
4. 计算候选路径代价和边界；越界则以 `cost_bound` 截断。
5. 调用 `get_successor_state()` 得到后继 `GlobalState`。
6. 计算后继的 `h` 或 dead-end 状态。
7. 使用原生 `test_goal()` 判断是否达到目标。
8. 在本 completion 的 visited map 中检查重复状态。
9. 记录逐状态结果。
10. 若达到目标，停止处理剩余动作，并将其标为 trailing actions。
11. 否则继续处理下一动作。

### 7.2 合法前缀定义

`valid_prefix_length` 表示从第一个动作开始，连续成功应用的动作数量：

- 非法动作本身不计入合法前缀。
- 达到目标的动作计入合法前缀。
- 目标之后未执行的动作不计入合法前缀，也不视为非法。
- 所有动作合法但未达到目标时，合法前缀长度等于生成动作总数。

“未达到目标”不等价于“存在非法动作”。必须区分：

- `goal_reached`
- `unknown_action`
- `inapplicable_action`
- `dead_end`
- `cost_bound`
- `end_of_sequence`，即全部合法但计划不完整

## 8. 状态身份与样本隔离

### 8.1 StateID 的含义

`StateRegistry` 按 packed state 内容做哈希和相等判断。不同动作路径到达相同世界状态时，会得到同一个进程内 `StateID`。

但必须注意：

- `StateID` 只在当前 task 和当前 C++ 进程生命周期内有效。
- 不同 scorer 副本的相同 `StateID` 没有可比性。
- 静态常量属于 task，不需要进入每个状态的身份。
- 路径 `g` 和部分 instrumentation/metric 信息可能不属于 packed world state，必须单独记录。
- 数值状态按 NLM 当前 packed 表示比较，不应在 Python 中重新实现另一套状态相等逻辑。

### 8.2 轨迹内新颖性

reward 使用的“新状态”必须定义为：

> 当前 completion 从初始状态开始模拟后，第一次出现的世界状态。

每条请求维护独立的：

```text
state_id -> first_seen_state_index
```

不得把“该状态是否第一次在长期 scorer 进程中注册”用于 reward，否则相同 completion 会因请求顺序不同得到不同分数。

应返回两个不同字段：

- `is_new_in_trajectory`：可以用于 reward。
- `is_new_in_registry`：只用于性能诊断，不能用于 reward。

### 8.3 重复状态与更优路径

如果后继 `StateID` 已在当前 completion 中出现：

- 记录 `is_repeated=true`。
- 返回 `first_seen_state_index`。
- 计算当前路径 `g` 是否小于此前到达该状态的最优 `g`。
- 若 `g` 更优，设置 `improved_g=true`，不要简单按普通循环惩罚。

对非负动作代价的普通规划问题，重复状态通常表示无效循环；但保留 `improved_g` 可以避免把特殊数值代价问题误判。

## 9. 请求协议

推荐使用 UTF-8 JSONL，一行一个完整请求。stdout 只写协议消息，所有日志写 stderr。

```json
{
  "type": "score_request",
  "protocol_version": 1,
  "request_id": "rollout-123",
  "problem_id": "problem_scale_10_id_1",
  "task_hash": "sha256:...",
  "actions": [
    "(lift hoist0 crate1 crate3 depot0)",
    "(load hoist0 crate1 truck0 depot0)"
  ],
  "options": {
    "return_states": true,
    "stop_at_goal": true
  }
}
```

必须设置请求大小、动作数量和单次评分时长上限，防止异常模型输出无限消耗 CPU 或内存。

## 10. 响应协议

### 10.1 总体结构

```json
{
  "type": "score_response",
  "protocol_version": 1,
  "request_id": "rollout-123",
  "problem_id": "problem_scale_10_id_1",
  "task_hash": "sha256:...",
  "status": "ok",
  "generated_action_count": 12,
  "valid_prefix_length": 7,
  "stop_reason": "inapplicable_action",
  "invalid_action_index": 7,
  "invalid_action": "(load ...)",
  "trailing_action_count": 4,
  "goal_reached": false,
  "goal_state_index": null,
  "start_h": 4.0,
  "final_h": 2.0,
  "best_h": 1.0,
  "best_h_state_index": 5,
  "unique_state_count": 6,
  "repeated_state_count": 1,
  "dead_end_reached": false,
  "path_cost": 3.0,
  "states": []
}
```

`invalid_action_index` 使用从 0 开始的动作下标。未发生非法动作时为 `null`。

### 10.2 逐状态列表

`states` 包含初始状态以及每个合法动作产生的后继状态，因此：

```text
len(states) = valid_prefix_length + 1
```

初始状态使用 `state_index=0`，没有 incoming action：

```json
{
  "state_index": 0,
  "state_id": 0,
  "state_label": "#0",
  "incoming_action_index": null,
  "incoming_action": null,
  "g": 0.0,
  "path_cost": 0.0,
  "h": 4.0,
  "h_status": "finite",
  "delta_h": 0.0,
  "relative_delta_h": 0.0,
  "is_new_best_h": true,
  "is_goal": false,
  "is_dead_end": false,
  "is_new_in_trajectory": true,
  "is_repeated": false,
  "first_seen_state_index": 0,
  "improved_g": false
}
```

后继状态示例：

```json
{
  "state_index": 1,
  "state_id": 17,
  "state_label": "#17",
  "incoming_action_index": 0,
  "incoming_action": "(lift hoist0 crate1 crate3 depot0)",
  "g": 1.0,
  "path_cost": 1.0,
  "h": 3.0,
  "h_status": "finite",
  "delta_h": 1.0,
  "relative_delta_h": 0.25,
  "is_new_best_h": true,
  "is_goal": false,
  "is_dead_end": false,
  "is_new_in_trajectory": true,
  "is_repeated": false,
  "first_seen_state_index": 1,
  "improved_g": false
}
```

字段定义：

- `delta_h = h_previous - h_current`，正数表示本步 h 改善。
- `relative_delta_h = delta_h / max(abs(start_h), h_scale_min)`。
- `is_new_best_h` 表示当前 h 严格优于此前轨迹最佳值，比较时使用配置的相对/绝对 epsilon。
- `h_status` 为 `finite` 或 `dead_end`。JSON 中不得使用 `Infinity`；dead end 时 `h=null`。
- `is_new_in_trajectory` 只与当前 completion 的 visited map 有关。
- `state_id` 仅用于当前响应内关联和调试。

生产训练中可以设置 `return_states=false`，只返回汇总字段以减少 IPC；抽样调试和离线分析时返回完整列表。

### 10.3 状态码

建议 `status` 取值：

- `ok`：评分成功，包括“全部合法但未到目标”。
- `parse_error`：Python 未能提取动作，通常不进入 C++。
- `invalid_request`：协议字段或 task hash 不合法。
- `task_mismatch`：请求 task hash 与进程加载任务不一致。
- `scorer_timeout`：评分超时。
- `internal_error`：NLM 内部错误。

规划语义上的 `unknown_action`、`inapplicable_action` 和 `dead_end` 应放在 `stop_reason`，不属于通信错误。

## 11. Reward 计算

### 11.1 设计原则

- 目标到达是最主要奖励，必须高于任何未完成轨迹可能获得的 shaping reward 上限。
- 合法动作长度不能单独无限加分，否则模型会通过生成冗长合法动作刷分。
- 重复状态应惩罚，避免重复动作 hacking reward。
- h 值允许局部不下降。`lmcutnumeric` 不是 consistent heuristic，必要动作可能暂时使 h 不变或上升。
- `h=0` 不代表目标到达，目标必须使用 NLM 原生 `test_goal()`。
- 不同问题的 h 量纲可能不同，必须进行相对归一化。
- reward 最终由 Python 计算并通过配置控制，C++ 不持有权重。

### 11.2 合法前缀奖励

定义：

```text
L_gen    = generated_action_count
L_valid  = valid_prefix_length
L_unique = 合法前缀中产生新轨迹状态的动作数
L_cap    = 配置的有效深度奖励上限
```

推荐：

```text
valid_fraction = L_valid / max(L_gen, 1)
unique_depth   = min(L_unique / max(L_cap, 1), 1)
prefix_quality = valid_fraction * unique_depth
r_prefix       = w_prefix * prefix_quality
```

这种形式同时要求动作大体合法且确实推进到新状态，并通过 `L_cap` 防止仅靠增加动作数持续获利。

特殊情况：

- 没有解析出动作：`r_prefix=0`，另加 parse penalty。
- 第一个动作非法：`L_valid=0`。
- 所有动作合法但未完成：可以获得部分 prefix reward，但不得接近 goal reward。
- 到达目标后存在多余输出：目标后的动作不影响合法前缀和非法动作统计，可另设轻微格式惩罚。

### 11.3 目标奖励

```text
r_goal = w_goal if goal_reached else 0
```

必须满足：

```text
w_goal > 所有非目标正向 shaping 项的最大可能总和
```

达到目标后可增加一个有界效率项：

```text
goal_efficiency = 1 / (1 + normalized_path_cost)
r_goal_efficiency = w_goal_efficiency * goal_efficiency
```

效率项不能大到使较短的失败轨迹优于较长的成功轨迹。

### 11.4 重复状态惩罚

定义：

```text
repeat_count = repeated_state_count
repeat_rate  = repeat_count / max(L_valid, 1)
```

推荐：

```text
r_repeat = -w_repeat * repeat_rate
```

还可以增加首次明显循环后的尾部惩罚：

```text
loop_tail = max(0, L_valid - first_repeat_action_index)
r_loop_tail = -w_loop_tail * min(loop_tail / L_cap, 1)
```

`improved_g=true` 的重复状态可以降低或免除循环惩罚。是否启用该例外由具体领域的动作代价性质决定。

### 11.5 启发式进展奖励

定义：

```text
h0      = start_h
hbest   = best_h
hfinal  = final_h
denom   = max(abs(h0), h_scale_min)
```

归一化：

```text
best_progress  = clip((h0 - hbest)  / denom, -1, 1)
final_progress = clip((h0 - hfinal) / denom, -1, 1)
```

推荐组合：

```text
r_h = w_best_h * best_progress + w_final_h * final_progress
```

`best_progress` 奖励曾经发现更有希望的状态，`final_progress` 防止模型先降低 h、随后通过循环退回坏状态仍获得全部进展奖励。

逐状态调试时可计算：

```text
new_best_gain_i = max(0, previous_best_h - h_i) / denom
```

这些增量之和与整体最佳进展相对应，但正式 sequence-level reward 不必把 reward 分配到每个动作 token。

dead end 时不能参与普通有限 h 运算，应单独计算：

```text
r_dead_end = -w_dead_end
```

### 11.6 非法动作与解析惩罚

建议区分：

```text
r_parse   = -w_parse    if parse_error
r_unknown = -w_unknown  if unknown_action
r_illegal = -w_illegal  if inapplicable_action
r_bound   = -w_bound    if cost_bound
```

惩罚可以结合非法动作位置：越早失败，惩罚越大。

```text
failure_position = invalid_action_index / max(L_gen, 1)
early_failure_factor = 1 - failure_position
```

### 11.7 初始总分公式

第一版可使用：

```text
reward =
    r_goal
  + r_goal_efficiency
  + r_prefix
  + r_h
  + r_repeat
  + r_loop_tail
  + r_parse
  + r_unknown
  + r_illegal
  + r_bound
  + r_dead_end
```

最终结果应裁剪到固定范围，例如：

```text
reward = clip(reward, reward_min, reward_max)
```

配置中必须保存全部权重、epsilon、`L_cap`、裁剪范围和 reward 版本。

## 12. veRL 返回值与日志

Reward manager 向 veRL 返回 scalar reward，同时附带 `reward_extra_info`：

```json
{
  "score": 0.42,
  "goal_reached": 0,
  "parse_ok": 1,
  "generated_action_count": 12,
  "valid_prefix_length": 7,
  "valid_fraction": 0.5833,
  "unique_state_count": 6,
  "repeated_state_count": 1,
  "start_h": 4.0,
  "best_h": 1.0,
  "final_h": 2.0,
  "best_progress": 0.75,
  "final_progress": 0.5,
  "stop_reason": "inapplicable_action",
  "scorer_seconds": 0.012
}
```

至少监控以下聚合指标：

- goal success rate
- parse success rate
- 平均合法前缀长度和比例
- 平均唯一状态比例
- loop/repeat rate
- dead-end rate
- `best_progress`、`final_progress`
- 各 `stop_reason` 占比
- scorer P50/P95/P99 延迟
- scorer cache hit rate
- scorer restart/crash/timeout 次数
- StateRegistry 状态数和 RSS

## 13. 性能注意事项

- 不在 reward 热路径运行 translator/preprocessor。
- 不使用 HTTP 作为本机默认协议；优先使用 stdin/stdout JSONL 或 Unix domain socket。
- 不等待同组样本组成 batch；允许多个独立请求异步并行。
- 对同一 task 的单个 C++ 进程先采用串行请求，避免线程安全问题。
- 通过增加 Ray scorer actors 或 task 副本扩展并行度。
- 默认响应只返回汇总；按比例抽样返回逐状态列表用于诊断。
- 设置最大动作数、最大 JSON 大小和评分超时。
- 记录 NLM 初始化耗时与纯轨迹评分耗时，分别判断 one-shot 和 stream 的收益。

第一版可以先使用 `--once` 测量。如果进程启动和任务初始化成本在总 reward 延迟中占比明显，再启用 `--stream` 和 LRU context cache；协议不需要改变。

## 14. 必须验证的边界情况

实现后至少覆盖以下测试：

1. 空回复和无法解析的回复。
2. 零动作回复。
3. 第一个动作不存在。
4. 第一个动作存在但不适用。
5. 中途出现非法动作，合法前缀下标正确。
6. 所有动作合法但未到目标，返回 `end_of_sequence`。
7. 中途达到目标，目标后的动作被忽略且不视为非法。
8. 初始状态已经满足目标。
9. 动作导致 dead end。
10. 两条不同动作路径到达相同状态。
11. 单条 completion 内形成循环，`first_seen_state_index` 正确。
12. 重复状态但路径 `g` 更优。
13. h 暂时上升后下降，不被错误截断。
14. h 为 0 但不是目标。
15. 数值 fluent 更新后 StateID、h 和目标判断正确。
16. 相同请求重复评分得到相同 reward。
17. 请求顺序变化不影响每条 completion 的 reward。
18. scorer 进程崩溃、超时和协议损坏时可以恢复。
19. task hash 不匹配时拒绝评分。
20. 长动作序列受到动作数和超时限制。

还应使用同一动作链对比：

- 当前 `inject_llm_action_chain()` 的逐步结果。
- 新 `ActionChainEvaluator` 的评分结果。
- 必要时 Unified Planning 的离线验证结果。

三者在合法前缀、最终状态和目标判断上应保持一致。

## 15. 建议的代码边界

后续实现可以按以下模块拆分：

```text
C++
  action_chain_evaluator.{h,cc}   纯动作链模拟与状态信息采集
  trajectory_score_protocol.*     请求/响应结构与 JSON 序列化
  trajectory_scorer_main.cc       --once / --stream 入口

Python
  reward/action_parser.py         LLM 动作函数块安全解析
  reward/nlm_client.py            C++ 子进程协议客户端
  reward/scorer_actor.py          Ray Actor、路由和 LRU 生命周期
  reward/reward_formula.py        可配置 reward 组合
  reward/verl_reward_manager.py   veRL RewardManager 适配
```

现有搜索代码应改为调用公共 `ActionChainEvaluator`，避免维护两套动作应用语义。搜索器仍负责 Open List 回插，评分器只读取公共模拟结果。

## 16. 实施阶段

### 阶段 A：纯 C++ 评分核心

- 从 `inject_llm_action_chain()` 提取公共动作模拟逻辑。
- 建立逐状态结果结构。
- 保证搜索回插行为不变。

### 阶段 B：one-shot 协议

- 实现 `nlm-score --once`。
- 完成协议、错误码和边界测试。
- 与现有回插日志逐步对齐。

### 阶段 C：veRL 接入

- 实现 Python parser、client 和 reward formula。
- 实现单样本异步 `NLMRewardManager.run_single()`。
- 先使用 one-shot scorer 获取正确性和延迟基线。

### 阶段 D：stream 与 Ray Actor 缓存

- 增加 `--stream`。
- 增加 scorer actor、task 路由和 LRU。
- 增加 RSS、状态数、请求数和 TTL 回收。

### 阶段 E：奖励消融与性能优化

- 分别测试 goal、prefix、repeat、h shaping 的贡献。
- 调整 reward 上下界和各项权重。
- 评估 scorer CPU 占用是否影响 rollout/training。
- 根据 P95 延迟决定 actor 数和热点 task 副本数。

## 17. 实现前仍需确定的配置

正式编码前需要冻结以下选择：

- veRL 具体版本及 RewardLoop/RewardManager 接口版本。
- 第一版是否直接实现 stream，还是先以 one-shot 建立基线。
- `lmcutnumeric` 的完整配置字符串。
- 是否把路径 metric 纳入 reward，以及 instrumentation 的候选内隔离方法。
- goal、prefix、repeat、h、dead-end 和非法动作的初始权重。
- 最大动作数、超时、最大 StateRegistry 状态数和 RSS。
- 是否默认返回逐状态列表，还是只对抽样请求返回。
- task cache 的生成脚本、路径规范和版本管理方式。

这些配置一旦用于正式训练，应连同 NLM commit、task hash、协议版本和 reward 版本一起写入实验元数据。

## 18. 相关实现与接口索引

当前仓库中与本设计直接相关的实现：

- [`EagerSearch::inject_llm_action_chain()`](../src/search/search_engines/eager_search.cc)：现有动作名称匹配、适用性检查、后继生成、启发式计算和 Open List 回插流程。
- [`StateRegistry`](../src/search/state_registry.h)：状态注册、内容哈希、重复状态识别和 `StateID` 管理。
- [`StateRegistry::get_successor_state()`](../src/search/state_registry.cc)：应用离散/数值效果、求值公理并注册后继状态。
- [`Heuristic::compute_result()`](../src/search/heuristic.cc)：启发式初始化、缓存和 dead-end 结果转换。
- [`test_goal()`](../src/search/globals.cc)：NLM 原生目标状态判断。
- [`response_processor.py`](../hybrid_planner/validation/response_processor.py)：现有模型动作块安全解析、Python action 名称转换和 UP 合法前缀验证。

veRL 侧后续接入时应以实际锁定版本的接口为准：

- [veRL Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)
- [veRL Extension Guide](https://verl.readthedocs.io/en/latest/extend_guide.html)

由于 veRL 接口仍在演进，正式实现前应把训练环境使用的 veRL commit 固定到实验配置中，不能只记录包版本名。
