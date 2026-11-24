"""事件日志与分析器的输入校验。"""
import unittest

from ckpt_consistency import EventLog, analyze


class ValidationTest(unittest.TestCase):
    def test_timestamps_must_be_monotonic(self):
        log = EventLog()
        log.step_start(t=10, step=1)
        with self.assertRaises(ValueError):
            log.grad_complete(t=9, step=1)

    def test_interval_end_must_not_precede_start(self):
        log = EventLog()
        with self.assertRaises(ValueError):
            log.tensor_write(8, 5, step=1, new_version=1, tensor="w1")

    def test_commit_without_reads_is_rejected(self):
        log = EventLog()
        log.checkpoint_commit(t=3, checkpoint="ghost", target_step=1)
        with self.assertRaises(ValueError):
            analyze(log)

    def test_duplicate_tensor_read_is_rejected(self):
        log = EventLog()
        log.checkpoint_read(1, 2, checkpoint="ckpt", tensor="w1")
        log.checkpoint_read(3, 4, checkpoint="ckpt", tensor="w1")
        log.checkpoint_commit(t=5, checkpoint="ckpt", target_step=1)
        with self.assertRaises(ValueError):
            analyze(log)


if __name__ == "__main__":
    unittest.main()
