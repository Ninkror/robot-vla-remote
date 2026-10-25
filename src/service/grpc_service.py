from __future__ import annotations

import threading
import time

import grpc
import numpy as np

from proto import remote_inference_pb2 as pb2
from proto import remote_inference_pb2_grpc as pb2_grpc
from router.manager import ModelUnavailable

from .validation import decode_rgb, validate_state


class RemoteInferenceService(pb2_grpc.RemoteInferenceServicer):
    def __init__(self, manager, codec_sha256: str, service_version: str, allow_motion: bool = False):
        self.manager = manager
        self.codec_sha256 = codec_sha256
        self.service_version = service_version
        self.allow_motion = bool(allow_motion)
        self._inference_lock = threading.Lock()

    @staticmethod
    def _model_info(value: dict):
        return pb2.ModelInfo(
            model_id=value["model_id"],
            display_name=value["display_name"],
            framework=value["framework"],
            status=value["status"],
            can_infer=bool(value["can_infer"]),
            can_execute=bool(value["can_execute"]),
            message=value["message"],
            active=bool(value["active"]),
        )

    def Health(self, request, context):  # noqa: N802
        active = self.manager.active_model_id
        ready = False
        device = ""
        if active:
            info = self.manager.info(active)
            ready = info["status"] in {"trained_dry_run", "validated"}
            device = self.manager.registry.get(active).device
        return pb2.HealthResponse(
            ready=ready,
            active_model_id=active,
            device=device,
            service_version=self.service_version,
            codec_sha256=self.codec_sha256,
        )

    def ListModels(self, request, context):  # noqa: N802
        return pb2.ListModelsResponse(
            models=[self._model_info(item) for item in self.manager.list_models()],
            default_model_id=self.manager.registry.default_model_id,
            active_model_id=self.manager.active_model_id,
        )

    def GetModelStatus(self, request, context):  # noqa: N802
        try:
            return self._model_info(self.manager.info(request.model_id))
        except KeyError as error:
            context.abort(grpc.StatusCode.NOT_FOUND, str(error))

    def _failure_response(self, request, code: str, message: str, received_ns: int):
        return pb2.PredictResponse(
            sequence_id=request.sequence_id,
            server_receive_time_unix_ns=received_ns,
            server_response_time_unix_ns=max(time.time_ns(), received_ns + 1),
            camera_mapping="single_camera_camera1",
            motion_allowed=False,
            status_code=code,
            status_message=message,
            codec_sha256=self.codec_sha256,
            model_id=request.model_id,
            model_status="loading" if code == "MODEL_LOADING" else "failed",
            session_id=request.session_id,
        )

    def Predict(self, request, context):  # noqa: N802
        received_ns = time.time_ns()
        if request.camera_mapping != "single_camera_camera1":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "camera_mapping must be single_camera_camera1")
        if not request.task.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "task must be non-empty")
        if not request.model_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_id is required")
        try:
            state = validate_state(request.state)
            image = decode_rgb(
                request.image.height,
                request.image.width,
                request.image.encoding,
                request.image.row_stride,
                request.image.data,
            )
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        try:
            spec, info = self.manager.ensure_ready(request.model_id, request.session_id)
        except ModelUnavailable as error:
            return self._failure_response(request, error.code, str(error), received_ns)
        if not self._inference_lock.acquire(blocking=False):
            return self._failure_response(
                request, "INFERENCE_BUSY", "an inference request is already active", received_ns
            )
        try:
            result = self.manager.predict(spec, image, state, request.task)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:
            context.abort(grpc.StatusCode.INTERNAL, f"model inference failed: {error}")
        finally:
            self._inference_lock.release()

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
            context.abort(grpc.StatusCode.INTERNAL, "runner returned an invalid action chunk")
        if result.get("codec_sha256") != self.codec_sha256:
            context.abort(grpc.StatusCode.INTERNAL, "worker action codec does not match the router")
        motion_allowed = bool(request.request_execution and self.allow_motion and spec.can_execute)
        if motion_allowed:
            try:
                self.manager.authorize_session(request.session_id, spec.model_id)
            except (ModelUnavailable, ValueError) as error:
                code = error.code if isinstance(error, ModelUnavailable) else "INVALID_SESSION"
                return self._failure_response(request, code, str(error), received_ns)
        return pb2.PredictResponse(
            sequence_id=request.sequence_id,
            action_rows=actions.shape[0],
            action_cols=actions.shape[1],
            actions=actions.reshape(-1).tolist(),
            inference_seconds=float(result["inference_seconds"]),
            server_receive_time_unix_ns=received_ns,
            server_response_time_unix_ns=max(time.time_ns(), received_ns + 1),
            model_path=str(result["model_path"]),
            camera_mapping="single_camera_camera1",
            motion_allowed=motion_allowed,
            status_code="OK",
            status_message=(
                "ACTION AUTHORIZED: client may execute this validated action chain"
                if motion_allowed
                else "DRY RUN: model output is not authorized for execution"
            ),
            codec_sha256=self.codec_sha256,
            model_id=spec.model_id,
            model_status=info["status"],
            session_id=request.session_id,
        )

    def ReleaseSession(self, request, context):  # noqa: N802
        return pb2.ReleaseSessionResponse(released=self.manager.release_session(request.session_id))
