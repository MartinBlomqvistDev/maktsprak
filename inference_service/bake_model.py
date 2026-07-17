"""Download the classifier into the image at build time.

Run by the Dockerfile, never at runtime. The service used to resolve the model
from the Hugging Face Hub on every cold start, which with scale-to-zero meant a
live demo depended on HF answering at the moment a visitor arrived. It stopped
answering (429 Too Many Requests), the container could not start, and Cloud Run
served 503. Baking the weights in removes the Hub from the request path
entirely and pins the deployment to the exact weights it was built with.
"""

from __future__ import annotations

import os
import sys
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "MartinBlomqvist/maktsprak_classifier_clean")
DEST = "/opt/model"
ATTEMPTS = 5


def main() -> None:
    # The Hub rate-limits (this is why the model is being baked in at all), so a
    # transient 429 during the build must not fail the build.
    for attempt in range(1, ATTEMPTS + 1):
        try:
            AutoModelForSequenceClassification.from_pretrained(MODEL_ID).save_pretrained(DEST)
            AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(DEST)
            print(f"baked {MODEL_ID} into {DEST}")
            return
        except Exception as exc:
            if attempt == ATTEMPTS:
                print(f"failed to fetch {MODEL_ID} after {ATTEMPTS} attempts: {exc}")
                sys.exit(1)
            backoff = 2**attempt * 5
            print(f"attempt {attempt}/{ATTEMPTS} failed ({exc}); retrying in {backoff}s")
            time.sleep(backoff)


if __name__ == "__main__":
    main()
