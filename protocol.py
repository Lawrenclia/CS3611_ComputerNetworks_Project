import struct

PAYLOAD_SIZE = 1024
DATA_HEADER_FORMAT = "!Id"
ACK_FORMAT = "!i"
DATA_HEADER_SIZE = struct.calcsize(DATA_HEADER_FORMAT)
ACK_SIZE = struct.calcsize(ACK_FORMAT)
DATA_PACKET_SIZE = DATA_HEADER_SIZE + PAYLOAD_SIZE


def build_payload(seq: int) -> bytes:
    marker = struct.pack("!I", seq)
    repeat = PAYLOAD_SIZE // len(marker)
    return marker * repeat


def pack_data_packet(seq: int, timestamp: float, payload: bytes) -> bytes:
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError(f"payload must be {PAYLOAD_SIZE} bytes")
    return struct.pack(DATA_HEADER_FORMAT, seq, timestamp) + payload


def unpack_data_packet(packet: bytes) -> tuple[int, float, bytes]:
    if len(packet) < DATA_HEADER_SIZE:
        raise ValueError("packet too small")
    seq, timestamp = struct.unpack(DATA_HEADER_FORMAT, packet[:DATA_HEADER_SIZE])
    payload = packet[DATA_HEADER_SIZE:]
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError("unexpected payload size")
    return seq, timestamp, payload


def pack_ack(ack_number: int) -> bytes:
    return struct.pack(ACK_FORMAT, ack_number)


def unpack_ack(packet: bytes) -> int:
    if len(packet) != ACK_SIZE:
        raise ValueError("invalid ack size")
    (ack_number,) = struct.unpack(ACK_FORMAT, packet)
    return ack_number
