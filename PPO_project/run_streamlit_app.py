"""一键启动可视化界面的入口脚本"""
import os
import sys
from pathlib import Path


def main():
    # 当前文件位于 PPO_project/ 下，实际的 app.py 同目录。
    # 若被移动到其它位置，则尝试父级目录。
    root = Path(__file__).resolve().parent
    app_path = root / "app.py"
    if not app_path.exists():
        app_path = root.parent / "PPO_project" / "app.py"
    if not app_path.exists():
        print(f"未找到可视化入口：{app_path}")
        sys.exit(1)

    # 等价于: streamlit run PPO_project/app.py
    try:
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(app_path)])
    except Exception as exc:  # pragma: no cover
        print(f"启动失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
