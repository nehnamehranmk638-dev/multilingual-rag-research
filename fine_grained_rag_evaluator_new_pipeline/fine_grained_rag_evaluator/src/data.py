import json

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_corpus(rows):
    return [{"id": str(x["id"]), "text": x["text"]} for x in rows]

def get_reference_claims(row):
    claims = row.get("reference_claims")
    return claims if isinstance(claims, list) else []
