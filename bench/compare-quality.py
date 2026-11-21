"""Capture matched generations and next-token probabilities, or compare two captures."""
import argparse
import difflib
import json
import math
import statistics
import urllib.request
from collections import Counter
from pathlib import Path

# Each reference is checkable without using another language model as a judge.
QUESTIONS = [
    ("arithmetic", "A store has 48 jars. It sells three eighths of them and receives 7 more. How many jars are there now? Reply with the number only.", "37"),
    ("time", "A train leaves at 22:47 and travels for 98 minutes. Give the arrival time on a 24-hour clock in HH:MM only.", "00:25"),
    ("Portuguese", "Responda apenas com o nome da pessoa. A Nara chegou antes do Ivo. O Raul chegou depois do Ivo. Quem chegou por último?", "Raul"),
    ("Python", "What does Python print? Reply only with the printed integer.\nx = [2, 5, 8, 11]\nprint(sum(x[::-2]))", "16"),
    ("SQL", "Table t has the values 2, NULL, 2, 7 in column k. Give the integer returned by SELECT COUNT(DISTINCT k) FROM t. No other text.", "2"),
    ("ordering", "Use these dependencies: B follows A, D follows C, C follows B. Write the unique order as four adjacent capital letters, without spaces.", "ABCD"),
    ("extraction", "Shipment MS-208 is pending. Shipment LS-410 arrived damaged. Shipment MS-280 arrived intact. Return just the ID of the damaged shipment.", "LS-410"),
    ("calendar", "If 1 March is a Tuesday, on which day of the week is 19 March? Respond with one English weekday.", "Saturday"),
    ("insufficient evidence", "A chart gives 18 visits on Monday and 26 on Wednesday. Tuesday's data are missing. Can the exact three-day total be determined? Answer yes or no only.", "no"),
    ("unit conversion", "A pump delivers 750 mL every 30 seconds. How many liters does it deliver in 8 minutes? Give just the number.", "12"),
    ("text transformation", "Change only the second occurrence of light to dark: light blue, light green, light gray. Return the resulting phrase only.", "light blue, dark green, light gray"),
    ("JSON", "Inventory has 2 red bolts and 5 blue bolts. Return a JSON object mapping red and blue to the counts as integers. No additional text.", {"red": 2, "blue": 5}),
]
PASSAGES = [
    "During the repair, a watchmaker found a folded receipt behind the dial. It was dated",
    "An index can speed up a selective database query because",
    "As nuvens já cobriam a serra quando o barco",
    "One reason a river can erode the outside of a bend is that",
    "def first_unique(items):\n    \"\"\"Return the first item that appears once, or None.\"\"\"\n",
    "The field journal described a bird with a chipped beak. On the following morning,",
]
CONTEXTS = [
    ("When setting up a telescope on a cold night,", " give the optics time to reach the temperature of the surrounding air."),
    ("The shortest path through an unweighted graph", " can be found by exploring each level with a queue."),
    ("The bread had been in the oven for twenty minutes when", " the smell of rosemary began to fill the kitchen."),
    ("O trem saiu da estação antes que", " os últimos passageiros chegassem à plataforma."),
    ("Hun lagde nøglerne på bordet og", " lukkede døren stille bag sig."),
    ("def flatten(rows):\n    return [", "value for row in rows for value in row]"),
    ("SELECT customer_id, COUNT(*)\nFROM orders\n", "GROUP BY customer_id\nORDER BY COUNT(*) DESC"),
    ("On a logarithmic scale, equal distances represent", " equal ratios rather than equal differences."),
    ("To rewind a cassette that has jammed,", " use a pencil and turn the spool gently."),
    ("A gardener puts three bulbs in each of seven pots. She uses", " twenty-one bulbs in total."),
    ("Jules checked the tide table, folded the map and", " walked down the slipway to the waiting boat."),
    ("The model card lists separate checksums for", " the weights and the image projection file."),
]
SETTINGS = [(0, 117), (1, 42), (1, 813)]


def normalized(s):
    return " ".join(s.casefold().strip().strip('."* ' + chr(96)).split())


def correct(answer, expected):
    if isinstance(expected, dict):
        try:
            obj = json.loads(answer.strip())
        except ValueError:
            return False
        return isinstance(obj, dict) and all(type(v) is int for v in obj.values()) and obj == expected
    return normalized(answer) == normalized(expected)


def capture(url, model, key_file, reference_inputs):
    headers = {"Content-Type": "application/json"}
    if key_file:
        headers["Authorization"] = "Bearer " + key_file.read_text().strip()

    def api(path, **payload):
        req = urllib.request.Request(url.rstrip("/") + "/" + path, json.dumps(dict(model=model, **payload)).encode(), headers)
        with urllib.request.urlopen(req, timeout=300) as f:
            return json.load(f)

    def encode(text, add):
        return api("tokenize", content=text, add_special=add)["tokens"]

    rows, answers, texts = [], [], []
    if reference_inputs:
        source = json.loads(reference_inputs.read_text())
        contexts = [(v["case"], v["pos"], v["ids"], v["teacher"]) for v in source["rows"]]
        passage_ids = [v["prompt"] for v in source["texts"]]
    else:
        contexts = []
        for case, (prefix, continuation) in enumerate(CONTEXTS):
            ids = encode(prefix, True)
            for pos, teacher in enumerate(encode(continuation, False)[:8]):
                contexts.append((case, pos, list(ids), teacher))
                ids.append(teacher)
        passage_ids = [encode(passage, True) for passage in PASSAGES]
    for case, pos, ids, teacher in contexts:
        out = api("completion", prompt=ids, n_predict=1, n_probs=20, temperature=0,
                  post_sampling_probs=False, return_tokens=True, cache_prompt=False, seed=117)
        row = out["completion_probabilities"][0]
        top = {str(t["id"]): math.exp(t["logprob"]) for t in row["top_logprobs"]}
        assert len(top) == 20 and 0 < sum(top.values()) <= 1.00002 and out["tokens"] == [row["id"]]
        rows.append(dict(case=case, pos=pos, ids=ids, teacher=teacher, choice=row["id"], top=top))
    for temp, seed in SETTINGS:
        sample = dict(temperature=temp, top_k=0, top_p=1., min_p=0., seed=seed)
        for name, prompt, expected in QUESTIONS:
            out = api("v1/chat/completions", messages=[dict(role="user", content=prompt)], max_tokens=768,
                      chat_template_kwargs={"reasoning_effort": "none"}, reasoning_effort="none", **sample)
            choice = out["choices"][0]
            answer = choice["message"].get("content") or ""
            answers.append(dict(name=name, temp=temp, seed=seed, answer=answer, correct=correct(answer, expected),
                                finish=choice["finish_reason"], tokens=out["usage"]["completion_tokens"]))
        if temp == 0:
            for ids in passage_ids:
                out = api("completion", prompt=ids, n_predict=80, return_tokens=True, cache_prompt=False, **sample)
                assert 0 < len(out["tokens"]) <= 80 and len(out["tokens"]) == out["tokens_predicted"]
                texts.append(dict(prompt=ids, tokens=out["tokens"], text=out["content"]))
    assert len(rows) == 95 and len(answers) == 36 and len(texts) == 6
    return dict(model=model, rows=rows, answers=answers, texts=texts)


def compare(a, b):
    matches, lo, hi = [], [], []
    for x, y in zip(a["rows"], b["rows"], strict=True):
        assert all(x[k] == y[k] for k in ("case", "pos", "ids", "teacher"))
        p, q = x["top"], y["top"]
        shared = p.keys() & q.keys()
        # Collapse the unreported vocabulary to get an observable TV lower bound.
        difference = [p[k] - q[k] for k in shared]
        lo.append((sum(map(abs, difference)) + abs(sum(difference))) / 2)
        hi.append(1 - sum(min(p[k], q[k]) for k in shared))
        matches.append(x["choice"] == y["choice"])
    print(f"Next-token top-1 agreement {sum(matches)}/{len(matches)}, mean total variation between {statistics.mean(lo):.4f} and {statistics.mean(hi):.4f}.")
    for temp in (0, 1):
        pairs = [(x, y) for x, y in zip(a["answers"], b["answers"], strict=True) if x["temp"] == temp]
        assert all((x["name"], x["temp"], x["seed"]) == (y["name"], y["temp"], y["seed"]) for x,y in pairs)
        changes = Counter((x["correct"], y["correct"]) for x, y in pairs)
        agree = sum(normalized(x["answer"]) == normalized(y["answer"]) for x, y in pairs)
        for side in (0,1):
            counts = Counter(x[side]["finish"] for x in pairs)
            print(("Reference" if side == 0 else "Candidate"), "finish reasons", dict(counts))
        print(f"T={temp}, reference {sum(x['correct'] for x,y in pairs)}/{len(pairs)}, candidate {sum(y['correct'] for x,y in pairs)}/{len(pairs)}, normalized exact answer agreement {agree}/{len(pairs)}, new errors {changes[True, False]}, newly correct {changes[False, True]}.")
        for x,y in pairs:
            if not x["correct"] or not y["correct"]:
                print(f"  {x['name']} seed {x['seed']}: {x['answer'][:180]!r} / {y['answer'][:180]!r}")
    lengths, overlaps = [], []
    for x,y in zip(a["texts"], b["texts"], strict=True):
        assert x["prompt"] == y["prompt"]
        u,v = x["tokens"], y["tokens"]
        lengths.append(next((i for i,(j,k) in enumerate(zip(u,v)) if j!=k), min(len(u),len(v))))
        overlaps.append(difflib.SequenceMatcher(None,u,v,autojunk=False).ratio())
    print("Greedy continuation common-prefix lengths", lengths, "of up to 80. Mean token sequence similarity", round(statistics.mean(overlaps),4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("capture")
    run.add_argument("url")
    run.add_argument("model")
    run.add_argument("output", type=Path)
    run.add_argument("--key-file", type=Path)
    run.add_argument("--reference-inputs", type=Path, help="Feed the reference token IDs exactly")
    compare_args = modes.add_parser("compare")
    compare_args.add_argument("reference", type=Path)
    compare_args.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        result = capture(args.url, args.model, args.key_file, args.reference_inputs)
        args.output.write_text(json.dumps(result, ensure_ascii=False))
    else:
        compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
