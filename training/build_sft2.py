import csv, json, random
API = "/mnt/workspace/lm/data/sft/raw/train.csv"
IDN = "/mnt/workspace/lm/data/sft/identity_1k.jsonl"
OUT = "/mnt/workspace/lm/data/sft/mix2.jsonl"

rows = []
with open(API, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        q = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        a = (r.get("output") or "").strip()
        if not q or not a or len(a) < 30 or len(a) > 900: continue
        if inp: q = q + "\n" + inp
        rows.append({"messages": [{"role":"user","content":q},{"role":"assistant","content":a}]})
rng = random.Random(11); rng.shuffle(rows)
al = rows[:3000]
idn = [json.loads(l) for l in open(IDN, encoding="utf-8")]
rng.shuffle(idn); idn = idn[:1000]
mix = al + idn; rng.shuffle(mix)
with open(OUT, "w", encoding="utf-8") as f:
    for x in mix: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"alpaca {len(al)}  identity {len(idn)}  total {len(mix)}")
