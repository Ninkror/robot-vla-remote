from __future__ import annotations

import argparse
from concurrent import futures
import hashlib
from pathlib import Path
import signal

import grpc

from common.paths import CONFIGS
from proto import remote_inference_pb2_grpc as pb2_grpc
from router.manager import WorkerManager
from router.registry import ModelRegistry

from .grpc_service import RemoteInferenceService


SERVICE_VERSION = "2.0.0"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def build_grpc_server(manager, codec_sha256: str, bind: str, allow_motion: bool = False):
    if not bind.startswith("127.0.0.1:"):
        raise ValueError("gRPC service must bind to the IPv4 loopback address")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=(
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ),
        maximum_concurrent_rpcs=8,
    )
    pb2_grpc.add_RemoteInferenceServicer_to_server(
        RemoteInferenceService(manager, codec_sha256, SERVICE_VERSION, allow_motion), server
    )
    port = server.add_insecure_port(bind)
    if port == 0:
        raise RuntimeError(f"failed to bind gRPC service to {bind}")
    return server, port


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Loopback-only multi-model VLA router")
    result.add_argument("--gpu", default=None)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8081)
    result.add_argument("--registry", type=Path, default=CONFIGS / "models.json")
    result.add_argument("--codec", type=Path, default=CONFIGS / "action_codec.json")
    result.add_argument("--actions", type=int, default=72)
    result.add_argument("--allow-motion", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.host != "127.0.0.1":
        raise ValueError("only --host=127.0.0.1 is permitted")
    registry = ModelRegistry(args.registry, gpu=args.gpu)
    codec_sha256 = hashlib.sha256(args.codec.read_bytes()).hexdigest()
    manager = WorkerManager(registry, actions_per_chunk=args.actions, codec_path=args.codec)
    try:
        try:
            manager.ensure_ready(registry.default_model_id, "")
        except Exception:
            pass
        server, port = build_grpc_server(
            manager, codec_sha256, f"{args.host}:{args.port}", args.allow_motion
        )
        server.start()

        def request_stop(_signum, _frame):
            server.stop(grace=5.0)

        signal.signal(signal.SIGTERM, request_stop)
        print(
            f"ROBOT_VLA_ROUTER_READY bind={args.host}:{port} "
            f"default_model={registry.default_model_id} "
            f"allow_motion={str(args.allow_motion).lower()}",
            flush=True,
        )
        server.wait_for_termination()
    finally:
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
