import argparse
from pathlib import Path, PurePath
from typing import Optional

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def segment(
    ecg,
    rpeaks,
    before: int = 300,
    after: int = 300,
    sampling_rate: int = 500,
    show: bool = False,
) -> pd.DataFrame:
    segmented = {}
    for i, peak in enumerate(rpeaks):
        start = peak - before
        end = peak + after
        if start >= 0 and end <= ecg.shape[-1]:
            segmented[i] = ecg[start:end]
            if type(segmented[i]) is pd.Series:
                segmented[i].reset_index(drop=True, inplace=True)
    segmented = pd.DataFrame(segmented)
    if show:
        plt.figure()
        X = np.arange(before + after)
        X = (X - (before + after) / 2) / sampling_rate
        for seg in segmented:
            plt.plot(X, segmented[seg])
        plt.title("Segmented beats")
        plt.xlabel("time (s)")
        plt.ylabel("signal")
        plt.xlim(-before / sampling_rate, after / sampling_rate)
        plt.grid()
        plt.show()
    return segmented


def clean_and_segment(
    ecg,
    show: bool = False,
    use_raw_if_error: bool = False,
    clean: Optional[list[str]] = None,
):
    SAMPLE_RATE = 500
    ecg = nk.ecg_invert(ecg, sampling_rate=SAMPLE_RATE)[0]
    rpeaks = None
    if clean is None:
        clean = []
    for method in clean:
        try:
            cleaned = nk.ecg_clean(ecg, sampling_rate=SAMPLE_RATE, method=method)
            rpeaks = nk.ecg_peaks(
                cleaned, sampling_rate=SAMPLE_RATE, method="neurokit"
            )[1]["ECG_R_Peaks"]
            assert len(rpeaks) > 0
        except Exception as e:
            continue
        break
    else:
        cleaned = ecg
        rpeaks = nk.ecg_peaks(cleaned, sampling_rate=SAMPLE_RATE, method="neurokit")[1][
            "ECG_R_Peaks"
        ]
    segmented = segment(cleaned, rpeaks, show=show)
    return cleaned, segmented


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and segment heart beats")
    parser.add_argument("-i", "--input", help="a file listing paths to each sample")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument(
        "--segment",
        "-s",
        action="store_true",
        help="output all segments of heart beats",
    )
    parser.add_argument(
        "--median",
        "-m",
        action="store_true",
        help="output median heart beat for all leads",
    )
    parser.add_argument(
        "--lead",
        "-l",
        action="store_true",
        help="output median heart beat for each lead",
    )
    args = parser.parse_args()
    return args


def main():
    old_header = pd.read_csv("header_info.csv", index_col="Filename")
    # old_header.drop(columns=["Unnamed: 0"], inplace=True)
    new_header = pd.DataFrame(columns=old_header.columns)
    # new_header.index.name = "Filename"

    status = {"valid rate": ""}
    total = 0
    valid = 0

    file_list = tqdm(list(Path("./npy").glob("**/*[0-9].npy")), postfix=status)
    for file in file_list:
        ecg = np.load(file)
        median = []
        for lead in ecg:
            try:
                _, segmented = clean_and_segment(lead)
                assert segmented.shape[0] == 600, segmented.shape
            except Exception as e:
                file_list.write("ERROR:\t" + str(file) + "\t" + repr(e))
                break
            med = segmented.median(axis=1)
            assert med.shape == (600,)
            median.append(med)
        else:
            valid += 1
            filename = file.name[:-4]
            np.save(
                PurePath.joinpath(file.parent, filename + "_median.npy"),
                np.array(median),
            )
            row = old_header.loc[filename]
            new_header.loc[filename + "_median"] = row
        total += 1
        status["valid rate"] = f"{valid}/{total} - {int(valid * 100 / total)}%"
        file_list.set_postfix(status, refresh=False)
    new_header.to_csv("header_median.csv", index_label="Filename")


# def main():
#     args = parse_args()
#     if args.input is None:
#         for i in Path(".").iterdir():
#             if i.suffix == ".txt":
#                 args.input = i
#     if args.output is None:
#         args.output = "segmented"
#     indir = PurePath.joinpath(Path.cwd(), args.input)
#     outdir = PurePath.joinpath(Path.cwd(), args.output, "csv")
#     outdir.mkdir(exist_ok=True, parents=True)
#     input(f"Reading from `{indir}` to `{outdir}`")
#
#     if (not args.lead) and (not args.median):
#         return
#
#     with open(indir) as file_list:
#         file_list = tqdm(list(file_list))
#         with open(PurePath.joinpath(outdir.parent, "valid.txt"), "w") as valid:
#             for xml_file in file_list:
#                 xml_file = Path(xml_file.strip())
#                 name = xml_file.name.split(".xml")[0]
#                 in_path = PurePath.joinpath(indir.parent, "csv/" + name + ".csv")
#                 data = pd.read_csv(in_path)
#
#                 df = {}
#
#                 for lead in LEADS:
#                     try:
#                         _, segmented = clean_and_segment(data[lead])
#                         assert segmented.shape[0] == 600, segmented.shape
#                     except Exception as e:
#                         file_list.write("ERROR:\t" + repr(e))
#                         break
#
#                     med = segmented.median(axis=1)
#                     df[lead] = med
#
#                 else:
#                     df = pd.DataFrame(df)
#                     if args.median:
#                         df["median"] = df.median(axis=1)
#                     if not args.lead:
#                         df = df[["median"]]
#                     df.to_csv(
#                         PurePath.joinpath(outdir.parent, "csv", name + ".csv"),
#                         index=False,
#                     )
#                     valid.write(str(xml_file) + "\n")


if __name__ == "__main__":
    main()
