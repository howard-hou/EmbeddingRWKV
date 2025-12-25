import sys
import json
from tabulate import tabulate

results = json.load(open(sys.argv[1], "r", encoding="utf-8"))
txt_path = sys.argv[1].replace(".json", ".txt")

main_scores = []
for r in results:
    try:
        score = r["scores"]["test"][0]["main_score"]
    except (KeyError, IndexError):
        score = None
    main_scores.append((r.get("task_name", "unknown"), score))

if main_scores:
    if tabulate:
        table_str = tabulate(main_scores, headers=["task_name", "main_score"], tablefmt="github")
    else:
        width = max(len(t[0]) for t in main_scores)
        lines = [f"{'task_name':<{width}}  main_score"]
        for task, score in main_scores:
            lines.append(f"{task:<{width}}  {score}")
        table_str = "\n".join(lines)
    print(table_str)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")