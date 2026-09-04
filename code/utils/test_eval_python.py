import builtins
import shutil
import subprocess
import unittest
from unittest.mock import patch

from eval_python import RUNNER_IMAGE, eval_python


class EvalPythonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("docker") is None:
            raise unittest.SkipTest("Docker is not installed")
        image = subprocess.run(
            ["docker", "image", "inspect", RUNNER_IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if image.returncode != 0:
            raise unittest.SkipTest(f"Sandbox image {RUNNER_IMAGE!r} is not built")

    def test_executes_fenced_candidate(self):
        code = """```python
def add(left, right):
    return left + right
```"""

        result = eval_python(code, "assert add(2, 3) == 5")

        self.assertEqual(0, result["exit_code"])

    def test_candidate_cannot_mutate_parent_builtins(self):
        attribute = "PROMPTINTERN_SANDBOX_TEST"
        code = f"import builtins\nbuiltins.{attribute} = True"

        result = eval_python(code, "")

        self.assertEqual(0, result["exit_code"])
        self.assertFalse(hasattr(builtins, attribute))

    def test_reports_candidate_exception(self):
        result = eval_python("raise ValueError('expected')", "")

        self.assertEqual(1, result["exit_code"])
        self.assertIn("ValueError: expected", result["error"])

    def test_runs_as_unprivileged_user_on_read_only_root(self):
        code = "import os\nresult = [os.getuid(), os.access('/', os.W_OK)]"

        result = eval_python(code, "")

        self.assertEqual(0, result["exit_code"])
        self.assertEqual([10001, False], result["output"])

    def test_has_no_network(self):
        code = (
            "import socket\n"
            "sock = socket.socket()\n"
            "sock.settimeout(1)\n"
            "result = sock.connect_ex(('1.1.1.1', 53))"
        )

        result = eval_python(code, "")

        self.assertEqual(0, result["exit_code"])
        self.assertNotEqual(0, result["output"])

    def test_limits_output_and_removes_container(self):
        result = eval_python("import os\nos.write(3, bytes(70000))", "")

        self.assertEqual(125, result["exit_code"])
        self.assertEqual([], self._sandbox_containers())

    def test_times_out_non_terminating_candidate(self):
        with patch("eval_python.EVALUATION_TIMEOUT_SECONDS", 1):
            result = eval_python("while True: pass", "")

        self.assertEqual(124, result["exit_code"])
        self.assertEqual([], self._sandbox_containers())

    @staticmethod
    def _sandbox_containers():
        containers = subprocess.run(
            [
                "docker", "ps", "--all", "--quiet",
                "--filter", "label=promptintern.sandbox=true",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        return containers.stdout.split()


if __name__ == "__main__":
    unittest.main()