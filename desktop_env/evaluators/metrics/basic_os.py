def check_gnome_favorite_apps(apps_str: str, rule):
    apps = eval(apps_str)

    expected_apps = rule["expected"]

    if len(apps) != len(expected_apps):
        return 0

    if set(apps) == set(expected_apps):
        return 1
    else:
        return 0


def is_utc_0(timedatectl_output):

    utc_line = timedatectl_output.split("\n")[3]

    if utc_line.endswith("+0000)"):
        return 1
    else:
        return 0


def check_text_enlarged(scaling_factor_str):
    scaling_factor = float(scaling_factor_str)
    if scaling_factor > 1.0:
        return 1
    else:
        return 0


def check_moved_jpgs(directory_list, rule):
    expected_jpgs = rule["expected"]
    moved_jpgs = [node['name'] for node in directory_list['children']]

    if len(moved_jpgs) != len(expected_jpgs):
        return 0

    if set(moved_jpgs) == set(expected_jpgs):
        return 1
    else:
        return 0


def is_in_vm_clickboard(config, terminal_output):
    print("terminal_output: ")
    print(terminal_output)
    print("config: ")
    print(config)
    expected_results = config["expected"]
    if not isinstance(expected_results, list):
        return 1 if expected_results in terminal_output else 0
    else:
        return 1 if all(result in terminal_output for result in expected_results) else 0
