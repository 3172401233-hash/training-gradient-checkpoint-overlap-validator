"""恢复链：可提交检查点推进恢复点，损坏检查点不推进，并给出重算范围。"""
import unittest

from ckpt_consistency import (
    COMMITTABLE,
    MIXED_VERSION,
    PARTIAL_WRITE,
    analyze,
)
from ckpt_consistency.scenarios import build_recovery_log


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.result = analyze(build_recovery_log())

    def test_good_checkpoint_is_committed_at_single_version(self):
        report = self.result.report_for("ckpt-good")
        self.assertEqual(report.status, COMMITTABLE)
        self.assertTrue(report.committable)
        self.assertEqual(report.consistent_version, 1)
        self.assertTrue(
            all(not obs.torn for obs in report.observations)
        )
        self.assertTrue(
            all(obs.observed_version == 1 for obs in report.observations)
        )
        self.assertIsNone(report.recompute)

    def test_mixed_checkpoint_rolls_back_to_last_good(self):
        report = self.result.report_for("ckpt-mixed")
        self.assertEqual(report.status, MIXED_VERSION)

        versions = sorted(
            obs.observed_version
            for obs in report.observations
            if obs.observed_version is not None
        )
        self.assertEqual(versions, [2, 3])

        recompute = report.recompute
        self.assertEqual(recompute.restore_checkpoint, "ckpt-good")
        self.assertEqual(recompute.restore_version, 1)
        self.assertEqual(recompute.from_step, 2)
        self.assertEqual(recompute.to_step, 3)
        self.assertEqual(recompute.step_count, 2)

    def test_torn_checkpoint_keeps_same_restore_point(self):
        report = self.result.report_for("ckpt-torn")
        self.assertEqual(report.status, PARTIAL_WRITE)

        recompute = report.recompute
        # 损坏的 ckpt-mixed 没有推进恢复点，回滚点仍是 ckpt-good@v1。
        self.assertEqual(recompute.restore_checkpoint, "ckpt-good")
        self.assertEqual(recompute.restore_version, 1)
        self.assertEqual(recompute.from_step, 2)
        self.assertEqual(recompute.to_step, 4)
        self.assertEqual(recompute.step_count, 3)

    def test_only_one_committable_checkpoint(self):
        good = self.result.committable_checkpoints
        self.assertEqual([r.checkpoint for r in good], ["ckpt-good"])
        self.assertEqual(
            [r.checkpoint for r in self.result.invalid_checkpoints],
            ["ckpt-mixed", "ckpt-torn"],
        )


if __name__ == "__main__":
    unittest.main()
