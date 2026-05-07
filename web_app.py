import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from protocol import PAYLOAD_SIZE, pack_ack, unpack_data_packet
from sender import ReliableSender


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UDP Reliable Transport Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --surface: #ffffff;
      --surface-2: #eef3f8;
      --text: #172033;
      --muted: #5f6b7a;
      --line: #d9e1ea;
      --accent: #007c89;
      --accent-2: #c2410c;
      --ok: #15803d;
      --warn: #b45309;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: var(--sans);
      color: var(--text);
      background: var(--bg);
    }

    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 32px;
    }

    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 20px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.18;
      font-weight: 760;
      letter-spacing: 0;
    }

    .subhead {
      margin: 0;
      max-width: 720px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      min-width: 270px;
    }

    .badge {
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }

    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 28px rgba(30, 42, 59, 0.08);
    }

    .panel h2 {
      margin: 0;
      padding: 16px 18px 0;
      font-size: 15px;
      letter-spacing: 0;
    }

    form {
      padding: 16px 18px 18px;
      display: grid;
      gap: 14px;
    }

    fieldset {
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: grid;
      gap: 12px;
    }

    legend {
      padding: 0 6px;
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.3;
    }

    input {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      color: var(--text);
      background: #fbfcfe;
      font: 14px var(--sans);
    }

    input:focus {
      outline: 2px solid rgba(0, 124, 137, 0.18);
      border-color: var(--accent);
    }

    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.68;
    }

    .pipeline {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      padding: 14px 18px 18px;
    }

    .stage {
      min-height: 74px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-2);
      padding: 10px;
      display: grid;
      align-content: center;
      gap: 5px;
    }

    .stage strong {
      font-size: 13px;
    }

    .stage span {
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
      overflow-wrap: anywhere;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 0 18px 18px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
      background: #fbfcfe;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }

    .metric strong {
      font-family: var(--mono);
      font-size: 18px;
      overflow-wrap: anywhere;
    }

    .logbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px 10px;
      border-top: 1px solid var(--line);
    }

    .status {
      color: var(--muted);
      font-size: 13px;
    }

    .status.ok { color: var(--ok); }
    .status.warn { color: var(--warn); }

    pre {
      margin: 0;
      min-height: 340px;
      max-height: 520px;
      overflow: auto;
      border-top: 1px solid var(--line);
      background: #101827;
      color: #d6e2ff;
      padding: 14px 18px;
      font: 12px/1.55 var(--mono);
      border-radius: 0 0 8px 8px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    @media (max-width: 900px) {
      header { display: block; }
      .badge-row { justify-content: flex-start; margin-top: 14px; }
      .layout { grid-template-columns: 1fr; }
      .pipeline { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 520px) {
      .shell { width: min(100vw - 20px, 1180px); padding-top: 18px; }
      h1 { font-size: 22px; }
      .metrics { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>UDP Reliable Transport Console</h1>
        <p class="subhead">本地回环测试面板：网页提交参数，后端启动 UDP 接收线程并调用可靠发送端，返回 ACK、RTO 与吞吐日志。</p>
      </div>
      <div class="badge-row" aria-label="protocol modules">
        <span class="badge">Header</span>
        <span class="badge">RTO</span>
        <span class="badge">ACK Thread</span>
        <span class="badge">Timer Thread</span>
      </div>
    </header>

    <section class="layout">
      <aside class="panel">
        <h2>测试参数</h2>
        <form id="runForm">
          <fieldset>
            <legend>Receiver 配置</legend>
            <label>
              监听地址
              <input name="receiver_host" value="127.0.0.1" required />
            </label>
            <label>
              监听端口，0 表示自动分配
              <input name="receiver_port" type="number" min="0" max="65535" value="0" required />
            </label>
            <label>
              初始期望序号
              <input name="receiver_initial_seq" type="number" min="0" max="2147483647" value="0" required />
            </label>
          </fieldset>

          <fieldset>
            <legend>Sender 配置</legend>
            <label>
              目标地址
              <input name="sender_target_host" value="127.0.0.1" required />
            </label>
            <label>
              本地地址
              <input name="sender_local_host" value="127.0.0.1" required />
            </label>
            <label>
              本地端口，0 表示自动分配
              <input name="sender_local_port" type="number" min="0" max="65535" value="0" required />
            </label>
            <label>
              起始序号
              <input name="sender_start_seq" type="number" min="0" max="2147483647" value="0" required />
            </label>
            <label>
              发送包数
              <input name="packets" type="number" min="0" max="500" value="12" required />
            </label>
            <label>
              滑动窗口
              <input name="window_size" type="number" min="1" max="128" value="4" required />
            </label>
            <label>
              RTO 秒
              <input name="rto" type="number" min="0.02" max="5" step="0.01" value="0.20" required />
            </label>
          </fieldset>
          <button id="runButton" type="submit">运行本地测试</button>
        </form>
      </aside>

      <section class="panel">
        <h2>协议链路</h2>
        <div class="pipeline" aria-label="transport pipeline">
          <div class="stage"><strong>Payload</strong><span>1024 bytes</span></div>
          <div class="stage"><strong>Header</strong><span>seq + timestamp</span></div>
          <div class="stage"><strong>UDP</strong><span>sendto / recvfrom</span></div>
          <div class="stage"><strong>Receiver</strong><span>expected_seq</span></div>
          <div class="stage"><strong>ACK</strong><span>signed cumulative ack</span></div>
        </div>

        <div class="metrics">
          <div class="metric"><span>确认进度</span><strong id="acked">-</strong></div>
          <div class="metric"><span>重传次数</span><strong id="retransmissions">-</strong></div>
          <div class="metric"><span>耗时</span><strong id="duration">-</strong></div>
          <div class="metric"><span>吞吐</span><strong id="throughput">-</strong></div>
        </div>

        <div class="logbar">
          <strong>运行日志</strong>
          <span id="status" class="status">等待运行</span>
        </div>
        <pre id="logs">点击“运行本地测试”后显示发送端与接收端日志。</pre>
      </section>
    </section>
  </main>

  <script>
    const form = document.querySelector("#runForm");
    const button = document.querySelector("#runButton");
    const statusEl = document.querySelector("#status");
    const logsEl = document.querySelector("#logs");
    const fields = {
      acked: document.querySelector("#acked"),
      retransmissions: document.querySelector("#retransmissions"),
      duration: document.querySelector("#duration"),
      throughput: document.querySelector("#throughput"),
    };

    function setStatus(text, kind) {
      statusEl.textContent = text;
      statusEl.className = "status" + (kind ? " " + kind : "");
    }

    function renderResult(data) {
      fields.acked.textContent = `${data.acked_packets}/${data.total_packets}`;
      fields.retransmissions.textContent = String(data.retransmissions);
      fields.duration.textContent = `${data.duration_s.toFixed(3)}s`;
      fields.throughput.textContent = `${data.throughput_mbps.toFixed(3)} Mbps`;
      logsEl.textContent = data.logs.join("\\n");
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = Object.fromEntries(new FormData(form).entries());
      payload.packets = Number(payload.packets);
      payload.window_size = Number(payload.window_size);
      payload.rto = Number(payload.rto);
      payload.receiver_port = Number(payload.receiver_port);
      payload.receiver_initial_seq = Number(payload.receiver_initial_seq);
      payload.sender_local_port = Number(payload.sender_local_port);
      payload.sender_start_seq = Number(payload.sender_start_seq);

      button.disabled = true;
      setStatus("运行中", "warn");
      logsEl.textContent = "正在启动本地 UDP 测试...";

      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || "request failed");
        }
        renderResult(data);
        setStatus("完成", "ok");
      } catch (error) {
        setStatus("失败", "warn");
        logsEl.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@dataclass
class TestReceiver:
    host: str
    port: int
    total_packets: int
    logs: list[str]
    initial_seq: int = 0
    ready: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    actual_port: int = 0
    unique_packets: int = 0
    total_bytes: int = 0
    thread: Optional[threading.Thread] = None
    startup_error: Optional[BaseException] = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=2.0):
            raise RuntimeError("receiver failed to start")
        if self.startup_error is not None:
            raise RuntimeError(f"receiver failed to start: {self.startup_error}")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
            sock.settimeout(0.2)
            self.actual_port = sock.getsockname()[1]
            self.ready.set()
            seen: set[int] = set()
            buffered: set[int] = set()
            expected_seq = self.initial_seq
            target_seq = self.initial_seq + self.total_packets
            self._log(f"listening on {self.host}:{self.actual_port} expected_seq={expected_seq}")

            while not self.stop_event.is_set() and expected_seq < target_seq:
                try:
                    packet, address = sock.recvfrom(2048)
                except socket.timeout:
                    continue

                try:
                    seq, timestamp, payload = unpack_data_packet(packet)
                except ValueError as exc:
                    self._log(f"ignore invalid packet: {exc}")
                    continue

                duplicate = seq in seen
                if not duplicate:
                    seen.add(seq)
                    self.total_bytes += len(payload)

                if seq < expected_seq:
                    status = "duplicate_or_late"
                elif seq == expected_seq:
                    status = "in_order"
                    expected_seq += 1
                    while expected_seq in buffered:
                        buffered.remove(expected_seq)
                        expected_seq += 1
                else:
                    status = "out_of_order"
                    if not duplicate:
                        buffered.add(seq)

                ack_number = expected_seq - 1
                sock.sendto(pack_ack(ack_number), address)
                self.unique_packets = len(seen)
                self._log(
                    "recv seq={seq} status={status} dup={dup} cum_ack={ack} "
                    "expected_seq={expected} buffered={buffered} ts={ts:.6f}".format(
                        seq=seq,
                        status=status,
                        dup=duplicate,
                        ack=ack_number,
                        expected=expected_seq,
                        buffered=len(buffered),
                        ts=timestamp,
                    )
                )
        except BaseException as exc:
            self.startup_error = exc
        finally:
            self.ready.set()
            sock.close()

    def _log(self, message: str) -> None:
        self.logs.append(f"[{time.strftime('%H:%M:%S')}][RECEIVER] {message}")


class CapturingSender(ReliableSender):
    def __init__(self, *args: Any, logs: list[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logs = logs

    def _log(self, category: str, message: str) -> None:
        if self.verbose:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}][SENDER][{category}] {message}")


class WebHandler(BaseHTTPRequestHandler):
    server_version = "ReliableTransportWeb/1.0"

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json({"ok": False, "error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = run_reliable_test(payload)
            self._send_json({"ok": True, **result})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def clamp_int(value: Any, minimum: int, maximum: int, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def clamp_float(value: Any, minimum: float, maximum: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def run_reliable_test(config: dict[str, Any]) -> dict[str, Any]:
    packets = clamp_int(config.get("packets", 12), 0, 500, "packets")
    window_size = clamp_int(config.get("window_size", 4), 1, 128, "window_size")
    rto = clamp_float(config.get("rto", 0.2), 0.02, 5.0, "rto")
    receiver_host = str(config.get("receiver_host", "127.0.0.1")).strip() or "127.0.0.1"
    receiver_port = clamp_int(config.get("receiver_port", 0), 0, 65535, "receiver_port")
    receiver_initial_seq = clamp_int(
        config.get("receiver_initial_seq", 0),
        0,
        2_147_483_647,
        "receiver_initial_seq",
    )
    sender_target_host = str(config.get("sender_target_host", receiver_host)).strip() or receiver_host
    sender_local_host = str(config.get("sender_local_host", "127.0.0.1")).strip() or "127.0.0.1"
    sender_local_port = clamp_int(config.get("sender_local_port", 0), 0, 65535, "sender_local_port")
    sender_start_seq = clamp_int(
        config.get("sender_start_seq", receiver_initial_seq),
        0,
        2_147_483_647,
        "sender_start_seq",
    )
    if sender_start_seq != receiver_initial_seq:
        raise ValueError("sender_start_seq must match receiver_initial_seq for cumulative ACK test")
    if sender_start_seq + max(packets - 1, 0) > 2_147_483_647:
        raise ValueError("sender_start_seq + packets exceeds signed ACK range")
    logs: list[str] = []

    receiver = TestReceiver(receiver_host, receiver_port, packets, logs, initial_seq=receiver_initial_seq)
    receiver.start()
    sender = CapturingSender(
        target_host=sender_target_host,
        target_port=receiver.actual_port,
        local_host=sender_local_host,
        local_port=sender_local_port,
        total_packets=packets,
        window_size=window_size,
        rto=rto,
        verbose=True,
        start_seq=sender_start_seq,
        logs=logs,
    )

    started_at = time.monotonic()
    sender_error: list[BaseException] = []

    def run_sender() -> None:
        try:
            sender.run()
        except BaseException as exc:
            sender_error.append(exc)

    try:
        sender_thread = threading.Thread(target=run_sender, daemon=True)
        sender_thread.start()
        timeout_s = max(5.0, packets * rto * 4.0 + 2.0)
        sender_thread.join(timeout=timeout_s)
        if sender_thread.is_alive():
            sender.stop_event.set()
            sender_thread.join(timeout=1.0)
            raise TimeoutError("sender did not finish before timeout")
        if sender_error:
            raise RuntimeError(f"sender failed: {sender_error[0]}")
    finally:
        receiver.stop()

    duration = max(time.monotonic() - started_at, 1e-6)
    throughput_mbps = (sender.acked_packets * PAYLOAD_SIZE * 8.0) / duration / 1_000_000.0

    return {
        "total_packets": packets,
        "acked_packets": sender.acked_packets,
        "retransmissions": sender.retransmissions,
        "duration_s": duration,
        "throughput_mbps": throughput_mbps,
        "sender_start_seq": sender_start_seq,
        "sender_local_host": sender_local_host,
        "sender_local_port": sender_local_port,
        "sender_target_host": sender_target_host,
        "receiver_host": receiver_host,
        "receiver_port": receiver.actual_port,
        "receiver_initial_seq": receiver_initial_seq,
        "receiver_unique_packets": receiver.unique_packets,
        "receiver_total_bytes": receiver.total_bytes,
        "logs": logs,
    }


def main() -> None:
    address = ("127.0.0.1", 8080)
    server = ThreadingHTTPServer(address, WebHandler)
    print(f"open http://{address[0]}:{address[1]}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
