"""Lifecycle management for the local vLLM OpenAI-compatible server."""

import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VLLMServiceConfig:
    """Settings adapted from PyPACE's ``qwen3_5_9B.bash`` deployment."""

    model_path: str = ""
    served_model_name: str = "Qwen3.5-9B"
    host: str = "127.0.0.1"
    port: int = 8091
    api_base_url: str = ""
    gpus: str = ""
    executable: str = "vllm"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    omp_num_threads: int = 2
    startup_timeout: float = 600.0
    poll_interval: float = 2.0
    log_path: str = "logs/vllm.log"
    extra_args: tuple = field(default_factory=tuple)

    @property
    def base_url(self):
        if self.api_base_url:
            return self.api_base_url.rstrip("/")
        return "http://%s:%d/v1" % (self.host, self.port)


class VLLMService:
    """Starts vLLM, waits for readiness, and owns its process and log."""

    def __init__(self, config):
        self.config = config
        self.process = None
        self._log_file = None

    def build_command(self):
        """Build the default ``vllm serve`` command without invoking a shell."""

        if not self.config.model_path:
            raise ValueError("vLLM model_path is required when launching the server")
        command = [
            self.config.executable,
            "serve",
            self.config.model_path,
            "--served-model-name",
            self.config.served_model_name,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--max-model-len",
            str(self.config.max_model_len),
            "--dtype",
            self.config.dtype,
        ]
        if self.config.trust_remote_code:
            command.append("--trust-remote-code")
        command.extend(self.config.extra_args)
        return command

    def start(self, command_override=None):
        """Launch vLLM and redirect both output streams to one log file."""

        if self.process is not None:
            return self.process
        command = list(command_override) if command_override else self.build_command()
        log_path = pathlib.Path(self.config.log_path).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path.open("ab")
        env = os.environ.copy()
        if self.config.gpus.strip():
            env["CUDA_VISIBLE_DEVICES"] = self.config.gpus
        env["OMP_NUM_THREADS"] = str(self.config.omp_num_threads)
        try:
            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self._log_file.close()
            self._log_file = None
            raise
        return self.process

    def _query_models(self):
        request = urllib.request.Request(
            "%s/models" % self.config.base_url.rstrip("/"),
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                return []
            body = json.loads(response.read().decode("utf-8"))
        return [item.get("id") for item in body.get("data", []) if item.get("id")]

    def wait_until_ready(self):
        """Poll ``/v1/models`` until the configured model is available."""

        deadline = time.monotonic() + self.config.startup_timeout
        last_error = "service has not answered yet"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "vLLM exited before becoming ready with code %s; see %s"
                    % (self.process.returncode, self.config.log_path)
                )
            try:
                models = self._query_models()
                if (
                    not self.config.served_model_name
                    or self.config.served_model_name in models
                ):
                    return models
                last_error = "available models are %r" % models
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(self.config.poll_interval)
        raise TimeoutError(
            "vLLM was not ready after %.1f seconds: %s"
            % (self.config.startup_timeout, last_error)
        )

    def stop(self):
        """Terminate the owned vLLM process and close its log."""

        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
