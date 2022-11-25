import math
import os
import json


def get_template_results(args):
    return {"config": args, "results": []}


def get_save_folder(model, date, label):
    year_month, day = date[:6], date[6:]
    week_number = math.ceil(int(day) / 7)

    folder_path = os.path.join(
        "results", "dev", model, year_month, "week_{}".format(week_number),
        "{}_{}".format(date, label)
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def save_results(results, label, model, date, run_id, seed):
    del results["config"]["device"]

    folder_path = get_save_folder(model, date, label)

    file_path = "run_{}_.json".format(seed)
    print('savepath', folder_path + ' ' + file_path)
    with open(os.path.join(folder_path, file_path), "w+") as f:
        try:
            json.dump(results, f, indent=2)
        except Exception:
            print("Failed to dump exps on json file.")
