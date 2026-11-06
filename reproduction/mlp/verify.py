"""Read-only integrity and CSV verification; standard library only."""
from pathlib import Path
import csv,hashlib,json,math
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    manifest=json.loads((HERE/"manifest.json").read_text())
    for entry in manifest["files"]:
        path=(HERE/entry["path"]).resolve()
        assert path.is_relative_to(HERE) and path.is_file(),entry["path"]
        assert digest(path)==entry["sha256"],entry["path"]
    protocol=json.loads((HERE/"artifacts/fullfit/protocol.json").read_text())
    identities=json.loads((HERE/"data_identity.json").read_text())["files"]
    for entry in identities:
        name=entry["path"]
        assert protocol["source_sha256"][name]==entry["source_sha256"]
        normalized=(ROOT/name).read_bytes().replace(b"\r\n",b"\n")
        assert hashlib.sha256(normalized).hexdigest()==entry["normalized_lf_sha256"],name
    with (ROOT/"data/example-submission.csv").open(newline="") as f:
        ids=[row["Id"] for row in csv.DictReader(f)]
    checked=[]
    for path in sorted((HERE/"artifacts").rglob("submission_mlp*.csv")):
        with path.open(newline="") as f:
            reader=csv.DictReader(f);assert reader.fieldnames==["Id","Predicted"]
            rows=list(reader)
        assert len(rows)==50000 and [r["Id"] for r in rows]==ids,path
        assert all(math.isfinite(float(r["Predicted"])) and float(r["Predicted"])>0 for r in rows),path
        checked.append({"file":str(path.relative_to(HERE)),"rows":len(rows),"sha256":digest(path)})
    scores=json.loads((HERE/"artifacts/fullfit/kaggle_results.json").read_text())
    for item in scores["results"]:
        if item["published_submission"]:
            assert digest(HERE/"artifacts/fullfit"/item["published_submission"])==item["local_submission_sha256"]
    assert len(checked)==4
    result={"files_verified":len(manifest["files"]),"submissions":checked,"source_csvs_unchanged":True,
            "kaggle_score_source":"Original user screenshots; weighted submissions have no observed score.","training_performed":False}
    print(json.dumps(result,indent=2));return result
if __name__=="__main__":verify()
