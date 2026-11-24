"""部分写入：读取区间与写入区间重叠即判定为撕裂读。"""
import unittest

from ckpt_consistency import PARTIAL_WRITE, EventLog, analyze
from ckpt_consistency.scenarios import build_partial_write_log


class PartialWriteTest(unittest.TestCase):
    def test_overlapping_read_is_torn(self):
        report = analyze(build_partial_write_log()).report_for("ckpt-b")

        self.assertEqual(report.status, PARTIAL_WRITE)
        self.assertFalse(report.committable)
        self.assertIsNone(report.consistent_version)

        by_tensor = {obs.tensor: obs for obs in report.observations}
        torn = by_tensor["w1"]
        clean = by_tensor["b1"]
        self.assertTrue(torn.torn)
        self.assertIsNone(torn.observed_version)
        self.assertEqual(torn.active_step, 2)
        self.assertIsNotNone(torn.conflicting_write)
        self.assertFalse(clean.torn)
        self.assertEqual(clean.observed_version, 2)

    def test_partial_write_forces_full_recompute_from_v0(self):
        report = analyze(build_partial_write_log()).report_for("ckpt-b")
        recompute = report.recompute

        self.assertIsNotNone(recompute)
        self.assertIsNone(recompute.restore_checkpoint)
        self.assertEqual(recompute.restore_version, 0)
        self.assertEqual(recompute.from_step, 1)
        self.assertEqual(recompute.to_step, 2)
        self.assertEqual(recompute.step_count, 2)

    def test_touching_intervals_are_not_torn_half_open_semantics(self):
        # 写入 [4, 8) 已在读取 [8, 10) 开始时完成，端点相接不算重叠。
        log = EventLog()
        log.tensor_write(4, 8, step=1, new_version=1, tensor="w1")
        log.checkpoint_read(8, 10, checkpoint="ckpt-edge", tensor="w1")
        log.checkpoint_commit(t=11, checkpoint="ckpt-edge", target_step=1)

        report = analyze(log).report_for("ckpt-edge")
        self.assertTrue(report.committable)
        observation = report.observations[0]
        self.assertFalse(observation.torn)
        self.assertEqual(observation.observed_version, 1)


if __name__ == "__main__":
    unittest.main()
