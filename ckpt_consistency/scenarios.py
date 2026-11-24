"""三组典型时间线，演示与单元测试共用。

张量只用名字（w1 权重、b1 偏置、m1 优化器动量状态）表示，
不存在真实张量数据。
"""
from .events import EventLog


def build_step_race_log():
    """step 竞争：读取横跨两个优化器版本，得到混合版本快照。

    step 3 更新 v3 时，w1 先写完、b1/m1 还没写；保存线程在这个
    窗口里读取，最终 w1=v3、b1=v2、m1=v2。
    """
    log = EventLog()
    # step 2 完整更新到 v2，作为竞争发生前的统一旧版本。
    log.step_start(t=0, step=2)
    log.grad_complete(t=1, step=2)
    log.opt_update_start(t=2, step=2, new_version=2)
    log.tensor_write(3, 4, step=2, new_version=2, tensor="w1")
    log.tensor_write(3, 4, step=2, new_version=2, tensor="b1")
    log.tensor_write(3, 4, step=2, new_version=2, tensor="m1")
    log.opt_update_end(t=5, step=2, new_version=2)

    log.step_start(t=10, step=3)
    log.grad_complete(t=12, step=3)
    log.opt_update_start(t=13, step=3, new_version=3)
    log.tensor_write(14, 18, step=3, new_version=3, tensor="w1")
    # w1 的 v3 写入 [14,18) 完成后再读，得到新版本 v3（无撕裂）。
    log.checkpoint_read(18, 19, checkpoint="ckpt-a", tensor="w1")
    # b1/m1 的 v3 写入尚未开始，读取拿到的还是上一版本 v2。
    log.checkpoint_read(18, 19, checkpoint="ckpt-a", tensor="b1")
    log.checkpoint_read(18, 19, checkpoint="ckpt-a", tensor="m1")
    log.tensor_write(20, 24, step=3, new_version=3, tensor="b1")
    log.tensor_write(22, 26, step=3, new_version=3, tensor="m1")
    log.opt_update_end(t=28, step=3, new_version=3)
    log.checkpoint_commit(t=30, checkpoint="ckpt-a", target_step=3)
    return log


def build_partial_write_log():
    """部分写入：读取与正在进行的写入区间重叠，读到写了一半的数据。"""
    log = EventLog()
    log.step_start(t=10, step=2)
    log.grad_complete(t=12, step=2)
    log.opt_update_start(t=13, step=2, new_version=2)
    log.tensor_write(14, 22, step=2, new_version=2, tensor="w1")
    log.checkpoint_read(18, 26, checkpoint="ckpt-b", tensor="w1")
    log.tensor_write(24, 28, step=2, new_version=2, tensor="b1")
    log.checkpoint_read(29, 31, checkpoint="ckpt-b", tensor="b1")
    log.opt_update_end(t=32, step=2, new_version=2)
    log.checkpoint_commit(t=34, checkpoint="ckpt-b", target_step=2)
    return log


def build_recovery_log():
    """恢复链：先有一个可提交检查点，后两个检查点分别损坏，
    验证回滚点不被损坏检查点推进，以及多 step 重算区间。"""
    log = EventLog()

    # step 1 完整结束后保存，干净且统一于 v1，可提交。
    log.step_start(t=0, step=1)
    log.grad_complete(t=2, step=1)
    log.opt_update_start(t=3, step=1, new_version=1)
    log.tensor_write(4, 6, step=1, new_version=1, tensor="w1")
    log.tensor_write(5, 7, step=1, new_version=1, tensor="b1")
    log.opt_update_end(t=8, step=1, new_version=1)
    log.checkpoint_read(9, 10, checkpoint="ckpt-good", tensor="w1")
    log.checkpoint_read(9, 10, checkpoint="ckpt-good", tensor="b1")
    log.checkpoint_commit(t=11, checkpoint="ckpt-good", target_step=1)

    # step 3 期间保存：w1 已到 v3，b1 仍是 v2，混合版本。
    log.step_start(t=20, step=2)
    log.grad_complete(t=22, step=2)
    log.opt_update_start(t=23, step=2, new_version=2)
    log.tensor_write(24, 26, step=2, new_version=2, tensor="w1")
    log.tensor_write(25, 27, step=2, new_version=2, tensor="b1")
    log.opt_update_end(t=28, step=2, new_version=2)

    log.step_start(t=30, step=3)
    log.grad_complete(t=32, step=3)
    log.opt_update_start(t=33, step=3, new_version=3)
    log.tensor_write(34, 38, step=3, new_version=3, tensor="w1")
    log.checkpoint_read(39, 40, checkpoint="ckpt-mixed", tensor="w1")
    log.checkpoint_read(39, 40, checkpoint="ckpt-mixed", tensor="b1")
    log.tensor_write(41, 44, step=3, new_version=3, tensor="b1")
    log.opt_update_end(t=46, step=3, new_version=3)
    log.checkpoint_commit(t=48, checkpoint="ckpt-mixed", target_step=3)

    # step 4 期间保存：w1 读取与写入重叠，部分写入，回滚点仍是 v1。
    log.step_start(t=50, step=4)
    log.grad_complete(t=52, step=4)
    log.opt_update_start(t=53, step=4, new_version=4)
    log.tensor_write(54, 60, step=4, new_version=4, tensor="w1")
    log.checkpoint_read(57, 62, checkpoint="ckpt-torn", tensor="w1")
    log.tensor_write(61, 65, step=4, new_version=4, tensor="b1")
    log.checkpoint_read(63, 64, checkpoint="ckpt-torn", tensor="b1")
    log.opt_update_end(t=66, step=4, new_version=4)
    log.checkpoint_commit(t=68, checkpoint="ckpt-torn", target_step=4)
    return log
