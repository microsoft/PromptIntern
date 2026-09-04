# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import re
import subprocess
import threading
import uuid

RUNNER_IMAGE = os.environ.get("PROMPTINTERN_RUNNER_IMAGE", "promptintern-python-runner:latest")
MAX_OUTPUT_BYTES = 64 * 1024
EVALUATION_TIMEOUT_SECONDS = 10


class SandboxCleanupError(RuntimeError):
    pass

def _extract_python(code):
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", code, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else code.strip()


def _read_bounded(stream, chunks, process, limit_reached):
    size = 0
    while data := stream.read(4096):
        size += len(data)
        if size > MAX_OUTPUT_BYTES:
            limit_reached.set()
            process.kill()
            break
        chunks.append(data)


def _run_sandbox(payload):
    container_name = f"promptintern-eval-{uuid.uuid4().hex}"
    command = [
        "docker", "run", "-i",
        "--name", container_name,
        "--label", "promptintern.sandbox=true",
        "--pull=never",
        "--user=10001:10001",
        "--network=none",
        "--ipc=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--memory-swap=256m",
        "--cpus=0.5",
        "--ulimit=nofile=64:64",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
        RUNNER_IMAGE,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_chunks = []
    stderr_chunks = []
    limit_reached = threading.Event()
    readers = [
        threading.Thread(
            target=_read_bounded,
            args=(stream, chunks, process, limit_reached),
            daemon=True,
        )
        for stream, chunks in (
            (process.stdout, stdout_chunks),
            (process.stderr, stderr_chunks),
        )
    ]
    for reader in readers:
        reader.start()

    try:
        process.stdin.write(json.dumps(payload).encode("utf-8"))
        process.stdin.close()
        process.wait(timeout=EVALUATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()
        process.stdout.close()
        process.stderr.close()
        try:
            cleanup = subprocess.run(
                ["docker", "rm", "--force", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            raise SandboxCleanupError(
                f"Could not remove sandbox container {container_name}."
            ) from error
        if cleanup.returncode != 0:
            raise SandboxCleanupError(
                f"Could not remove sandbox container {container_name}."
            )

    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    return process.returncode, stdout, stderr, limit_reached.is_set()


def eval_python(code, testcase):
    candidate = _extract_python(code)
    function = candidate + "\n" + testcase
    base_result = {"function": function, "input": testcase, "output": None}

    try:
        return_code, stdout, stderr, output_limited = _run_sandbox(
            {"code": candidate, "testcase": testcase}
        )
    except FileNotFoundError:
        return {
            **base_result,
            "exit_code": 127,
            "error": "Docker is required to run generated Python safely.",
        }
    except subprocess.TimeoutExpired:
        return {**base_result, "exit_code": 124, "error": "Evaluation timed out."}
    except SandboxCleanupError as error:
        return {**base_result, "exit_code": 126, "error": str(error)}

    if output_limited:
        return {**base_result, "exit_code": 125, "error": "Evaluation output limit exceeded."}
    if return_code != 0:
        error = stderr.strip()[-2000:] or "The sandbox exited without a result."
        return {**base_result, "exit_code": return_code, "error": error}

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return {**base_result, "exit_code": 1, "error": "The sandbox returned an invalid result."}
    return {**base_result, **result}

if __name__ == "__main__":
    code = '''
python```
def sort_matrix(M):\r\n    result = sorted(M, key=sum)\r\n    return result\n
```
'''
    testcase = '''
assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]
assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]
assert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]
'''

    print(eval_python(code,testcase))