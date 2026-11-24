"""step 竞争：保存窗口横跨两个优化器版本，产生混合版本快照。"""
import unittest

from ckpt_consistency import (
    INITIAL_VERSION,
    MIXED_VERSION,
    EventLog,
    analyze,
)
from ckpt_consistency.scenarios import build_step_race_log


class StepRaceTest(unittest.TestCase):
    def test_mixed_versions_across_update_boundary(self):
        result = analyze(build_step_race_log())

        self.assertEqual(len(result.reports), 1)
        report = result.report_for("ckpt-a")
        self.assertEqual(report.status, MIXED_VERSION)
        self.assertFalse(report.committable)
        self.assertIsNone(report.consistent_version)
        self.assertEqual(result.committable_checkpoints, [])

        by_tensor = {obs.tensor: obs for obs in report.observations}
        self.assertFalse(by_tensor["w1"].torn)
        self.assertEqual(by_tensor["w1"].observed_version, 3)
        self.assertEqual(by_tensor["w1"].active_step, 3)
        self.assertFalse(by_tensor["b1"].torn)
        self.assertEqual(by_tensor["b1"].observed_version, 2)
        self.assertEqual(by_tensor["m1"].observed_version, 2)

    def test_mixed_checkpoint_does_not_advance_restore_point(self):
        report = analyze(build_step_race_log()).report_for("ckpt-a")

        self.assertIsNotNone(report.recompute)
        recompute = report.recompute
        self.assertIsNone(recompute.restore_checkpoint)
        self.assertEqual(recompute.restore_version, INITIAL_VERSION)
        self.assertEqual(recompute.from_step, 1)
        self.assertEqual(recompute.to_step, 3)
        self.assertEqual(recompute.step_count, 3)

    def test_fully_before_update_reads_initial_version(self):
        log = EventLog()
        log.step_start(t=0, step=1)
        log.grad_complete(t=1, step=1)
        log.opt_update_start(t=2, step=1, new_version=1)
        log.tensor_write(5, 7, step=1, new_version=1, tensor="w1")
        log.tensor_write(6, 8, step=1, new_version=1, tensor="b1")
        log.checkpoint_read(3, 4, checkpoint="ckpt-0", tensor="w1")
        log.checkpoint_read(3, 4, checkpoint="ckpt-0", tensor="b1")
        log.opt_update_end(t=9, step=1, new_version=1)
        log.checkpoint_commit(t=10, checkpoint="ckpt-0", target_step=1)

        report = analyze(log).report_for("ckpt-0")
        self.assertTrue(report.committable)
        self.assertEqual(report.consistent_version, INITIAL_VERSION)
        self.assertIsNone(report.recompute)


if __name__ == "__main__":
    unittest.main()
