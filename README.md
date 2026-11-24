# training-gradient-checkpoint-overlap-validator

验证“训练反向传播尚未结束时异步保存检查点”的快照一致性。

## 背景问题

训练线程在反向传播结束后立即应用优化器更新，参数与优化器状态按张量逐个
非原子写入；保存线程若在更新进行中读取，就可能把**一部分新参数和一部分
旧优化器状态**写进同一个检查点。这样的检查点不对应任何单一优化器版本，
恢复后无法保证参数与优化器状态匹配。

本工具对一组按逻辑时钟排列的事件做纯逻辑分析，判定每个检查点：

- **可提交（committable）**：所有张量读取均已完成且观测到同一个优化器版本；
- **混合版本（mixed_version）**：读取均未被写入打断，但张量来自不同版本；
- **部分写入（partial_write）**：至少一次读取与正在进行的张量写入区间重叠，
  读到写了一半的数据。

不可提交的检查点不会推进恢复点；工具同时给出回滚位置和需要重放的
step 范围（重算范围）。

## 不加载真实模型

本仓库是**事件级模拟器**，不依赖 PyTorch/TensorFlow 等任何深度学习框架：

- 没有真实张量、参数、梯度或优化器状态，张量只用字符串名字（如 `w1`、`m1`）表示；
- 不启动真实保存线程，不做真实 I/O，所有“时间”都是整数逻辑时钟；
- 输入是 step、参数版本、梯度完成、优化器更新、检查点读取与提交事件，
  输出是快照一致性结论与重算范围。

## 运行环境

- Python 3.8.2（仅使用标准库 `dataclasses`、`typing`、`unittest`，无第三方依赖）。

## 目录结构

```
ckpt_consistency/
  events.py       七类事件与 EventLog（含时间戳/区间合法性校验）
  analyzer.py     逐张量判定撕裂读、混合版本，计算重算范围
  reporting.py    文本报告渲染
  scenarios.py    三组典型时间线（step 竞争 / 部分写入 / 恢复链）
  __main__.py     命令行演示
tests/            unittest 用例（step 竞争、部分写入、恢复、输入校验）
```

## 使用方式

运行演示时间线：

```bash
python -m ckpt_consistency
```

运行全部测试：

```bash
python -m unittest
```

也可以在代码里直接构造事件：

```python
from ckpt_consistency import EventLog, analyze, format_report

log = EventLog()
log.step_start(t=10, step=3)
log.grad_complete(t=12, step=3)
log.opt_update_start(t=13, step=3, new_version=3)
log.tensor_write(14, 18, step=3, new_version=3, tensor="w1")
log.checkpoint_read(18, 19, checkpoint="ckpt-a", tensor="w1")
log.opt_update_end(t=28, step=3, new_version=3)
log.checkpoint_commit(t=30, checkpoint="ckpt-a", target_step=3)

result = analyze(log)
print(format_report(result))
```

## 事件与判定语义

| 事件 | 含义 |
| --- | --- |
| `StepStart(t, step)` | 一个训练 step 开始 |
| `GradComplete(t, step)` | 反向传播/梯度计算完成 |
| `OptUpdateStart(t, step, new_version)` | 优化器更新开始 |
| `TensorWrite(t_start, t_end, step, new_version, tensor)` | 单张量非原子写入区间 |
| `OptUpdateEnd(t, step, new_version)` | 优化器更新整体结束，版本切换完成 |
| `CheckpointRead(t_start, t_end, checkpoint, tensor)` | 保存线程对单张量的读取区间 |
| `CheckpointCommit(t, checkpoint, target_step)` | 检查点尝试提交 |

- 区间采用半开语义 `[t_start, t_end)`：读取与写入仅在端点相接
  （`read.t_start == write.t_end`）时不算重叠，写入在读取开始前已完成。
- 读取区间与任意同张量写入区间真正重叠 => `partial_write`。
- 无重叠时，观测版本 = 读取开始前最后一个完成的写入版本；
  此前没有写入完成则为初始版本 `v0`。
- 全部张量干净且版本统一 => `committable`，检查点只对应这一个优化器版本；
  干净但版本不统一 => `mixed_version`。
- 恢复时回滚到最近一个可提交检查点（没有则回到初始 `v0`），
  重放 `restore_version + 1 .. target_step` 的全部 step。

## 测试覆盖

- **step 竞争**：读取横跨两个优化器版本，得到混合版本且不推进恢复点；
  以及更新开始前读取得到统一初始版本的对照用例。
- **部分写入**：读取与写入区间重叠判定为撕裂读，半开区间端点相接不算重叠。
- **恢复**：可提交检查点推进恢复点，后续混合/撕裂检查点不推进，
  多 step 重算区间正确。
- **输入校验**：点事件时间戳单调、区间合法、提交必须有读取、张量不可重复读。
