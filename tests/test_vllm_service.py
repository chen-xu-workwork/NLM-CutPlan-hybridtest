import json
import pathlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from hybrid_planner.llm.vllm_service import VLLMService, VLLMServiceConfig


class VLLMServiceTests(unittest.TestCase):
    def test_build_command_matches_runtime_deployment_contract(self):
        service = VLLMService(
            VLLMServiceConfig(
                model_path="/models/checkpoint",
                served_model_name="Qwen3.5-9B",
                port=8091,
                gpus="0",
            )
        )
        command = service.build_command()
        self.assertEqual(command[:3], ["vllm", "serve", "/models/checkpoint"])
        self.assertIn("--served-model-name", command)
        self.assertIn("Qwen3.5-9B", command)
        self.assertIn("--trust-remote-code", command)
        self.assertIn("--max-model-len", command)

    def test_waits_for_openai_models_endpoint(self):
        class ModelsHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                encoded = json.dumps(
                    {"data": [{"id": "Qwen3.5-9B"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), ModelsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            service = VLLMService(
                VLLMServiceConfig(
                    served_model_name="Qwen3.5-9B",
                    api_base_url=(
                        "http://127.0.0.1:%d/v1"
                        % server.server_address[1]
                    ),
                    startup_timeout=1,
                    poll_interval=0.01,
                )
            )
            self.assertEqual(
                service.wait_until_ready(),
                ["Qwen3.5-9B"],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_launch_failure_closes_log_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = VLLMService(
                VLLMServiceConfig(
                    model_path="/models/checkpoint",
                    log_path=str(pathlib.Path(temp_dir) / "vllm.log"),
                )
            )
            with mock.patch(
                "hybrid_planner.llm.vllm_service.subprocess.Popen",
                side_effect=OSError("launch failed"),
            ):
                with self.assertRaisesRegex(OSError, "launch failed"):
                    service.start()

            self.assertIsNone(service._log_file)
            self.assertIsNone(service.process)


if __name__ == "__main__":
    unittest.main()
