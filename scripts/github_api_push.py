"""Push a local git tree to GitHub via the Git Data API.

Fallback for sandboxed environments where github.com:443 is blocked but
api.github.com is allowed. Creates blobs -> tree -> commit -> branch ref,
so the remote history looks exactly like a normal `git push` of HEAD.
The token is read from the GITHUB_TOKEN environment variable, never stored.
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "wanzi-qz/wanzi-mitosam"
API = f"https://api.github.com/repos/{REPO}"
TOKEN = os.environ["GITHUB_TOKEN"]


def api(method: str, path: str, payload: dict | None = None, expect: int = 201):
    """Call the GitHub API with retry/backoff for transient 5xx from the proxy."""
    data = json.dumps(payload).encode() if payload is not None else None
    last: Exception | None = None
    for attempt in range(7):
        try:
            req = urllib.request.Request(f"{API}/{path}", data=data, method=method)
            req.add_header("Authorization", f"Bearer {TOKEN}")
            req.add_header("Accept", "application/vnd.github+json")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
                if r.status != expect:
                    raise RuntimeError(f"{method} {path} -> {r.status}")
                return body
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise  # 4xx = real problem, do not retry
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        wait = 2 ** attempt
        print(f"  retry {attempt + 1} for {method} {path} after {wait}s ({last})", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {method} {path}: {last}")


def main():
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    print(f"uploading {len(files)} files")

    # The Git Data API rejects blobs on an empty repository ("Git Repository
    # is empty", 409). Seed an initial commit through the Contents API first.
    try:
        api("GET", "git/ref/heads/main", expect=200)
        print("branch main already exists")
    except urllib.error.HTTPError as e:
        if e.code not in (404, 409):  # 404 no ref / 409 empty repository
            raise
        print("empty repo -> seeding initial commit via Contents API")
        api("PUT", "contents/README.md",
            {"message": "init", "content": base64.b64encode(b"MitoSAM\n").decode()},
            expect=201)

    parent = api("GET", "git/ref/heads/main", expect=200)["object"]["sha"]
    print(f"parent commit {parent[:10]}")

    entries = []
    for i, path in enumerate(files, 1):
        with open(path, "rb") as f:
            content = f.read()
        blob = api("POST", "git/blobs",
                   {"content": base64.b64encode(content).decode(), "encoding": "base64"})
        mode = "100755" if os.access(path, os.X_OK) and not path.endswith((".py", ".md", ".yaml", ".toml")) else "100644"
        entries.append({"path": path.replace("\\", "/"), "mode": mode, "type": "blob", "sha": blob["sha"]})
        print(f"  [{i}/{len(files)}] {path}", flush=True)

    tree = api("POST", "git/trees", {"tree": entries})
    print(f"tree {tree['sha'][:10]}")

    commit = api("POST", "git/commits", {
        "message": ("MitoSAM: mitochondria SR-image segmentation, quantification "
                    "and classification toolbox\n\n- SAM + CBAM/adapter fine-tuning pipeline "
                    "with point-prompt training\n- 130-dim feature families: morphology (28) / "
                    "texture (46) / spatial (23)\n- evaluation stack: repeated-CV ablation with "
                    "paired comparisons, label-permutation control, SHAP attribution, Otsu baseline\n"
                    "- latent-severity synthetic benchmark (60 FOVs shipped, scalable to 300)\n"
                    "- batch CLI + Gradio 3-tab GUI + one-command reproduction pipeline"),
        "tree": tree["sha"],
        "parents": [parent],
    })
    print(f"commit {commit['sha'][:10]}")

    ref = api("PATCH", "git/refs/heads/main", {"sha": commit["sha"], "force": True}, expect=200)
    print(f"ref main -> {ref['object']['sha'][:10]}")


if __name__ == "__main__":
    main()
