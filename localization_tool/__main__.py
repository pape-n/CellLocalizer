"""Entry point: python -m localization_tool [--self-test REPORT | --software-opengl | init ... | analyze ...]"""

import sys


def run(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1 and argv[1] in ("init", "analyze"):
        from localization_tool.core.cli import main as cli_main

        return cli_main(argv[1:])
    if "--self-test" in argv:
        from localization_tool.selftest import main as selftest_main

        i = argv.index("--self-test")
        report = argv[i + 1] if i + 1 < len(argv) else None
        return selftest_main(report)
    if "--software-opengl" in argv:
        # Remote Desktop or old graphics drivers: draw with the processor instead of the graphics card
        import os

        os.environ["QT_OPENGL"] = "software"
        argv = [a for a in argv if a != "--software-opengl"]
    from localization_tool.gui.app import run_gui

    if "--gui-smoke" in argv:
        # opens the window with the synthetic example, runs the performance test, writes the report, quits
        i = argv.index("--gui-smoke")
        report = argv[i + 1] if i + 1 < len(argv) else "gui_smoke.txt"
        return run_gui([argv[0]], smoke_report=report)
    return run_gui(argv)


if __name__ == "__main__":
    sys.exit(run())
