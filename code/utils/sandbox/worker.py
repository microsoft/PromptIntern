import json
import os
import sys
import traceback


def main():
    payload = json.load(sys.stdin)
    source = payload["code"] + "\n" + payload["testcase"]
    namespace = {"__name__": "__candidate__"}
    saved_stdout = os.dup(sys.stdout.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        exec(compile(source, "<candidate>", "exec"), namespace, namespace)
        response = {
            "exit_code": 0,
            "output": namespace.get(
                "result",
                "<The test case does not return an output>",
            ),
        }
    except BaseException:
        response = {
            "exit_code": 1,
            "output": None,
            "error": traceback.format_exc(limit=5)[-2000:],
        }
    finally:
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.close(saved_stdout)
        os.close(devnull)

    json.dump(response, sys.stdout, default=repr)


if __name__ == "__main__":
    main()