import argparse
import socket
import time

from protocol import pack_ack, unpack_data_packet


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}][RECEIVER] {message}", flush=True)


def run_receiver(host: str, port: int, initial_seq: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    seen = set()
    buffered = set()
    total_bytes = 0
    expected_seq = initial_seq
    log(f"listening on {host}:{port} expected_seq={expected_seq}")

    while True:
        packet, address = sock.recvfrom(2048)
        try:
            seq, timestamp, payload = unpack_data_packet(packet)
        except ValueError as exc:
            log(f"ignore invalid packet from {address}: {exc}")
            continue

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
    args = parser.parse_args()
    run_receiver(args.host, args.port, args.initial_seq)


if __name__ == "__main__":
    main()
