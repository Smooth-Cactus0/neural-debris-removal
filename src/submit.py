"""
submit.py — SSL-patched competition submission utility for this machine.

The Kaggle CLI's `competitions submit` fails because competition file uploads go through
Google Cloud Storage (googleapis.com) using `google-resumable-media`, which has its own SSL
stack unaffected by the kagglesdk patch.  This script patches both pathways before importing
anything that opens network connections.

Usage:
    python submit.py <csv_path> "<message>"
"""
import ssl

# Patch SSL before any other imports — must be first
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: E402

import sys
import urllib3
import requests

# Monkey-patch requests.Session.send to skip SSL verification (covers GCS upload)
_orig_send = requests.Session.send


def _send_no_verify(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)


requests.Session.send = _send_no_verify
urllib3.disable_warnings()

COMPETITION = "neural-debris-removal-in-streak-detection-models"


def submit(csv_path: str, message: str, competition: str = COMPETITION) -> None:
    import kaggle

    api = kaggle.api
    api.authenticate()
    api.competition_submit(
        file_name=csv_path,
        message=message,
        competition=competition,
        quiet=False,
    )
    print(f"Submitted '{csv_path}' -> {competition}")


def list_submissions(competition: str = COMPETITION) -> None:
    import kaggle

    api = kaggle.api
    api.authenticate()
    api.competition_submissions_cli(competition)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python submit.py <csv_path> <message>")
        sys.exit(1)
    submit(sys.argv[1], sys.argv[2])
