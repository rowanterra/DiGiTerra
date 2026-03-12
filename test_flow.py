#!/usr/bin/env python3
"""Quick test of the app flow: upload -> correlation -> preprocess.
Run from project root: python test_flow.py
Uses Flask test client; does not start a real server.
"""
import os
import sys

# Run from project root so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    from app import app
    client = app.test_client()
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "examples", "iris.csv")

    print("1. GET / ...")
    r = client.get("/")
    assert r.status_code == 200, f"GET / failed: {r.status_code}"
    print("   OK")

    print("2. POST /upload (iris.csv) ...")
    with open(csv_path, "rb") as f:
        r = client.post("/upload", data={"file": (f, "iris.csv")})
    assert r.status_code == 200, f"POST /upload failed: {r.status_code} {r.get_data(as_text=True)[:200]}"
    data = r.get_json()
    assert "numcols" in data, str(data)
    numcols = data["numcols"]
    print(f"   OK (numcols={numcols})")

    print("3. POST /correlationMatrices ...")
    # colsIgnore: 'all' or list of column indices to use (numeric only for correlation)
    r = client.post(
        "/correlationMatrices",
        json={
            "colsIgnore": "all",
            "dropMissing": "none",
            "imputeStrategy": "none",
            "dropZero": "none",
        },
        content_type="application/json",
    )
    assert r.status_code == 200, f"POST /correlationMatrices failed: {r.status_code} {r.get_data(as_text=True)[:300]}"
    print("   OK")

    print("4. POST /preprocess ...")
    # indicators = first 4 columns (0,1,2,3), predictors = last column (4), stratify = 0 (not target)
    r = client.post(
        "/preprocess",
        json={
            "indicators": [0, 1, 2, 3],
            "predictors": [4],
            "stratify": 0,
        },
        content_type="application/json",
    )
    assert r.status_code == 200, f"POST /preprocess failed: {r.status_code} {r.get_data(as_text=True)[:300]}"
    out = r.get_json()
    assert "predictors" in out and "indicators" in out, str(out)
    print(f"   OK (predictors={out.get('predictors')}, indicators count={len(out.get('indicators', []))})")

    print("\nAll steps completed successfully. Backend is not stalling.")


if __name__ == "__main__":
    main()
