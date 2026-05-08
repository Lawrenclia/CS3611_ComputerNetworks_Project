import argparse
import random
import socket
import time
from typing import Optional

from protocol import pack_ack, unpack_data_packet


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}][RECEIVER] {message}", flush=True)


def run_receiver(
    host: str,
    port: int,
    initial_seq: int,
    loss_rate: float = 0.0,
    delay_ms: float = 0.0,
    jitter_ms: float = 0.0,
    seed: Optional[int] = None,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    rng = random.Random(seed)
    seen = set()
    buffered = set()
    total_bytes = 0
    expected_seq = initial_seq
    log(
        "listening on {host}:{port} expected_seq={expected} loss_rate={loss:.3f} "
        "delay_ms={delay:.1f} jitter_ms={jitter:.1f}".format(
            host=host,
            port=port,
            expected=expected_seq,
            loss=loss_rate,
            delay=delay_ms,
            jitter=jitter_ms,
        )
    )

    while True:
        packet, address = sock.recvfrom(2048)
        try:
            seq, timestamp, payload = unpack_data_packet(packet)
        except ValueError as exc:
            log(f"ignore invalid packet from {address}: {exc}")
            continue

        if loss_rate > 0 and rng.random() < loss_rate:
            log(f"drop seq={seq} from={address[0]}:{address[1]} reason=simulated_loss")
            continue

        if delay_ms > 0 or jitter_ms > 0:
            jitter = rng.uniform(-jitter_ms, jitter_ms) if jitter_ms > 0 else 0.0
            delay_seconds = max(0.0, delay_ms + jitter) / 1000.0
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        duplicate = seq in seen
        if not duplicate:
            seen.add(seq)
            total_bytes += len(payload)

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
        ack = pack_ack(ack_number)
        sock.sendto(ack, address)
        log(
            "recv seq={seq} dup={dup} status={status} bytes={size} from={src} ts={ts:.6f} "
            "cum_ack={ack} expected_seq={expected} buffered={buffered} unique_packets={count} total_payload={total}".format(
                seq=seq,
                dup=duplicate,
                status=status,
                size=len(payload),
                src=f"{address[0]}:{address[1]}",
                ts=timestamp,
                ack=ack_number,
                expected=expected_seq,
                buffered=len(buffered),
                count=len(seen),
                total=total_bytes,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="UDP receiver for reliable transport demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--initial-seq", type=int, default=0)
    parser.add_argument("--loss-rate", type=float, default=0.0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.loss_rate <= 1.0:
        raise SystemExit("--loss-rate must be in [0, 1]")
    if args.delay_ms < 0:
        raise SystemExit("--delay-ms must be non-negative")
    if args.jitter_ms < 0:
        raise SystemExit("--jitter-ms must be non-negative")
    run_receiver(
        args.host,
        args.port,
        args.initial_seq,
        loss_rate=args.loss_rate,
        delay_ms=args.delay_ms,
        jitter_ms=args.jitter_ms,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
