#!/bin/zsh
set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "UDP 可靠传输与拥塞控制验证"
echo

python3 demo_runner.py
STATUS=$?

echo
if [ "$STATUS" -eq 0 ]; then
  echo "演示流程完成。结果已保存到 artifacts/demo_results 目录。"
else
  echo "演示流程出现错误，退出码: $STATUS"
fi

echo
echo "按任意键关闭窗口..."
read -k 1
exit "$STATUS"
