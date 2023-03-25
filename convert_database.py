import pathlib
import traceback
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

import ecg_segment


def main():
    # Read SNOMED-CT codes
    snomed = pd.read_csv("ConditionNames_SNOMED-CT.csv")["Snomed_CT"]

    # Create empty dataframe to store header info
    diagnoses = snomed.unique().tolist()
    header_info = pd.DataFrame(columns=["Age", "Sex"] + diagnoses)

    # Glob all .hea files in .
    files = list(pathlib.Path(".").glob("**/*.hea"))
    npy_dir = pathlib.Path("./npy")
    npy_dir.mkdir(exist_ok=True)
    for file in tqdm(files):
        try:
            if "ipy" in file.absolute().as_posix():
                continue
            # Read header
            record_name = file.stem
            # print("processing", file)
            file_without_ext = file.with_suffix("")
            # Read record
            signal = wfdb.rdrecord(file_without_ext)

            # Get ECG signal
            ecg = signal.p_signal.T

            # Save ecg to npy
            np.save(pathlib.PurePath.joinpath(npy_dir, record_name), ecg)
            # print("saved npy to", pathlib.PurePath.joinpath(npy_dir, record_name))

            header = wfdb.rdheader(file_without_ext)
            row: dict[Any, Any] = {d: 0 for d in diagnoses}
            row["Age"] = 0
            row["Sex"] = 0
            # Check if the record has a SNOMED-CT code
            for comment in header.comments:
                if comment.startswith("Age"):
                    try:
                        row["Age"] = int(comment.split(":")[1].strip())
                    except ValueError:
                        row["Age"] = -1
                elif comment.startswith("Sex"):
                    row["Sex"] = int(comment.split(":")[1].strip() == "Female")
                elif comment.startswith("Dx"):
                    # Get SNOMED-CT code
                    snomed_codes = comment.split(":")[1].strip().split(",")

                    # Get SNOMED-CT description
                    for code in snomed_codes:
                        try:
                            row[int(code.strip())] = 1
                        except KeyError:
                            continue
            header_info.loc[record_name] = row
        except Exception as e:
            print(e)
            print("error processing", file)
            traceback.print_exc()
            return
    header_info.to_csv("header_info.csv", index_label="Filename")


if __name__ == "__main__":
    main()
