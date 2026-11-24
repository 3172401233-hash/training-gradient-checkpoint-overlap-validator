"""命令行演示：python -m ckpt_consistency

只跑纯事件模拟，不加载真实模型。
"""
from .analyzer import analyze
from .reporting import format_report
from .scenarios import (
    build_partial_write_log,
    build_recovery_log,
    build_step_race_log,
)


def main():
    builders = [
        ("场景 1：step 竞争 -> 混合版本", build_step_race_log),
        ("场景 2：部分写入 -> 撕裂读", build_partial_write_log),
        ("场景 3：恢复链 -> 重算范围", build_recovery_log),
    ]
    blocks = []
    for title, builder in builders:
        result = analyze(builder())
        blocks.append(title + "\n" + format_report(result))
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
