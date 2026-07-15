"""
Score detector-vs-human agreement for the forced-choice scheme
(annotations.json: per prompt, the human-chosen least-hallucinatory model).

For each prompt the detector's pick = the model with the lowest score; we report
agreement with the human for two detector criteria (overall hallucination, and ER
alone), pooled and by task. A random baseline would be 1/(#models per prompt).

Usage: python human_eval/compute_kappa.py human_eval/annotations.json
"""
import json
import os
import sys
import collections


def main(path):
    ann = json.load(open(path))["annotations"]
    root = os.path.dirname(os.path.abspath(__file__))
    S = {s["uid"]: s for s in json.load(open(os.path.join(root, "samples.json")))}

    def argmin_model(item, key):
        best, bestv = None, 1e9
        for m in item["models"]:
            v = m["scores"][key]
            if v < bestv:
                bestv, best = v, m["model"]
        return best

    n = agree_ov = agree_er = 0
    rand = 0.0
    by_task = collections.defaultdict(lambda: [0, 0, 0])  # n, agree_ov, agree_er
    for uid, a in ann.items():
        if uid not in S or not a.get("chosen"):
            continue
        item = S[uid]
        human = a["chosen"]
        det_ov = argmin_model(item, "overall")
        det_er = argmin_model(item, "ER_factual_fabrication")
        n += 1
        rand += 1.0 / len(item["models"])
        ao = int(human == det_ov); ae = int(human == det_er)
        agree_ov += ao; agree_er += ae
        t = item["task"]
        by_task[t][0] += 1; by_task[t][1] += ao; by_task[t][2] += ae
    if not n:
        print("no annotations scored"); return
    print(f"prompts answered: {n}")
    print(f"human vs detector(min OVERALL): agree {agree_ov}/{n} = {100*agree_ov/n:.1f}%")
    print(f"human vs detector(min ER):      agree {agree_er}/{n} = {100*agree_er/n:.1f}%")
    print(f"random-pick baseline:           {100*rand/n:.1f}%")
    print("\nby task (agree% overall / ER):")
    for t, (nn, ao, ae) in sorted(by_task.items()):
        print(f"  {t:24} n={nn:4}  overall={100*ao/nn:.0f}%  ER={100*ae/nn:.0f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "human_eval/annotations.json")
