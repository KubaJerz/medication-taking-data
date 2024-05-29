import subprocess
import webbrowser
import signal
import os

processes = []

directories = [
    "../processed/3_final/03/2024-01-06_14_29_18", "../processed/3_final/03/2024-01-06_14_38_45",
    "../processed/3_final/03/2024-01-06_14_58_22", "../processed/3_final/03/2024-01-09_15_33_11",
    "../processed/3_final/03/2024-01-09_15_52_17", "../processed/3_final/03/2024-01-09_17_58_06",
    "../processed/3_final/03/2024-01-11_12_13_52", "../processed/3_final/03/2024-01-13_15_11_31",
    "../processed/3_final/03/2024-01-13_20_05_16", "../processed/3_final/03/2024-01-13_20_26_28"
]

port = 8051

for directory in directories:
    csv_file = f"{directory}/acceleration.csv"
    command = f"python plt_dash_with_doubble.py {csv_file} {port}"
    process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    processes.append(process)
    url = f"http://localhost:{port}"
    webbrowser.open_new_tab(url)
    port += 1

try:
    while True:
        pass
except KeyboardInterrupt:
    for process in processes:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)


# import subprocess
# import webbrowser
# import signal
# import os

# processes = []

# directories = [
#     "../processed/3_final/09/2023-01-09_11_13_55", "../processed/3_final/09/2023-01-09_11_23_08",
#     "../processed/3_final/09/2023-01-09_11_25_05", "../processed/3_final/09/2023-01-09_11_39_41",
#     "../processed/3_final/09/2023-01-09_11_46_36", "../processed/3_final/09/2023-01-09_15_47_22",
#     "../processed/3_final/09/2023-01-27_10_02_28", "../processed/3_final/09/2023-01-27_14_55_00",
#     "../processed/3_final/09/2023-01-31_08_50_07", "../processed/3_final/09/2023-01-31_12_13_08"
# ]

# port = 8051

# for directory in directories:
#     csv_file = f"{directory}/acceleration.csv"
#     command = f"python plt_dash_with_doubble.py {csv_file} {port}"
#     process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
#     processes.append(process)
#     url = f"http://localhost:{port}"
#     webbrowser.open_new_tab(url)
#     port += 1

# try:
#     while True:
#         pass
# except KeyboardInterrupt:
#     for process in processes:
#         os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# # import subprocess
# # import webbrowser

# # directories = [
# #     "./07/2024-02-07_11_39_39", "./07/2024-02-08_15_02_23", "./07/2024-02-09_13_32_52",
# #     "./07/2024-02-11_13_03_00", "./07/2024-02-11_13_07_05", "./07/2024-02-13_18_21_53",
# #     "./07/2024-02-14_11_07_03", "./07/2024-02-16_13_31_18", "./07/2024-02-18_13_02_35",
# #     "./07/2024-02-19_11_02_02", "./07/2024-02-21_11_07_13", "./07/2024-02-21_11_10_25"
# # ]

# # port = 8051

# # for directory in directories:
# #     csv_file = f"{directory}/acceleration.csv"
# #     command = f"python plt_dash_with_doubble.py {csv_file} {port}"
# #     process = subprocess.Popen(command, shell=True)
# #     url = f"http://localhost:{port}"
# #     webbrowser.open_new_tab(url)
# #     port += 1
