import subprocess
import mlflow

def log_git_commit():
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
            )
            .strip()
        )

        branch = (
            subprocess.check_output(
                ["git", "branch", "--show-current"],
                text=True,
            )
            .strip()
        )

        mlflow.set_tag("git_branch", branch)
        mlflow.set_tag("git_commit", commit)

    except Exception:
        pass