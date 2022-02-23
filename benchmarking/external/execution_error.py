from pathlib import Path


class ExecutionError(Exception):
    def __init__(self, message: str, executable_path: Path, exit_code: int, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.executable_path = executable_path
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def to_dict(self):
        return {
            "message": str(self),
            "executable": str(self.executable_path),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
