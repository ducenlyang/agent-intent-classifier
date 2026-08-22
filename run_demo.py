"""一键启动整个解决方案：意图网关(8601) + 对话后端(8600) + 打开网页Demo。

用法(仓库根目录): .venv\\Scripts\\python.exe run_demo.py
Ctrl+C 一并停止两个服务。
"""
from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
PY = sys.executable
GATEWAY = "http://127.0.0.1:8601"
BACKEND = "http://127.0.0.1:8600"


def wait_online(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    procs = [
        subprocess.Popen(
            [PY, "-m", "uvicorn", "intent_classifier.api:app",
             "--host", "127.0.0.1", "--port", "8601"],
            cwd=ROOT,
        ),
        subprocess.Popen(
            [PY, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", "8600"],
            cwd=ROOT / "edu-chat-backend",
        ),
    ]
    print("启动中: 意图网关(8601, 含tiny-bert预热约10秒) + 对话后端(8600)...")
    try:
        if wait_online(GATEWAY):
            print(f"✅ 意图网关在线 {GATEWAY}")
        else:
            print("⚠️ 意图网关启动超时，请查看其控制台输出")
        if wait_online(BACKEND):
            print(f"✅ 对话后端在线 {BACKEND}")
            print("正在打开网页Demo...")
            webbrowser.open(BACKEND)
        else:
            print("⚠️ 对话后端启动超时，请查看其控制台输出")
        print("\n两个服务运行中，Ctrl+C 停止全部。\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止全部服务...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        print("已全部停止。")


if __name__ == "__main__":
    main()
