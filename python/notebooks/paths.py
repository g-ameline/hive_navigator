from pipe import (
    Pipe,
     where,
    chain
)
import os

@Pipe
def first_two_lines_from_filepath(filepath):
    with open(filepath, mode='r') as stream:
        return stream.readline().strip(), stream.readline().strip()
    
hive_name_from_number = lambda number: f'hive_{number:02d}'

data_folderpath = "../../data/"
audio_folder = "audio/"
sensors_folder = "sensors/"
hive_foldernames = [(hive_name_from_number(number)+'/') for number in range(1,12)]
features_folderpath = data_folderpath + "features/"

hive_sensors_filename_from_hive_number = lambda hive_number:f'hive_{hive_number:02d}_sensors.csv'
hive_accelerometries_filename_from_hive_number = lambda hive_number:f'hive_{hive_number:02d}_accel.csv'

@Pipe
def hive_audio_folderpath_from_hive_number(hive_number):
    assert 1<=hive_number<=11
    return (
        data_folderpath 
        + audio_folder 
        + hive_foldernames[hive_number-1] 
        + "03/"
    )

@Pipe
def hive_sensors_filepath_from_hive_number(hive_number):
    assert 1<=hive_number<=11
    return (
        data_folderpath 
        + sensors_folder 
        + hive_foldernames[hive_number-1] 
        + "03/"
        + hive_sensors_filename_from_hive_number(hive_number)
    )

@Pipe
def hive_accelerometry_filepath_from_hive_number(hive_number):
    assert 1<=hive_number<=11
    return (
        data_folderpath 
        + sensors_folder 
        + hive_foldernames[hive_number-1] 
        + "03/"
        + hive_accelerometries_filename_from_hive_number(hive_number)
    )


def sorted_filepaths_from_folderpath(folderpath):
    return sorted(
        (
            os.path.join(subfolderpath, filename)
            for subfolderpath, _, filenames in os.walk(folderpath)
            for filename in filenames
        ),
        key=os.path.basename,
    )

def metadata_from_filepath(filepath):
    filename = os.path.basename(filepath)
    assert filename.endswith(".flac"), filename
    stem = filename.removesuffix(".flac")
    match stem.split("_"):
        case [prefix, hive_number, date, time]:
            return prefix + "_" + hive_number, date + "_" + time
        case _:
            assert False, stem


def audio_features_filename_from_hive_number(number):
    return f'hive_{number:02d}_audio_features.csv'
    
ambient_features_filename = "ambient_features.csv"
ambient_features_filepath = features_folderpath + "ambient_features.csv"
    
def all_audio_features_csv_filepaths(hive_numbers=range(1,7)):
    return (
        (
            (features_folderpath + audio_features_filename_from_hive_number(hive_number))
            # for hive_number in range(1,12)
            for hive_number in hive_numbers
        )
        | where(lambda filepath: os.path.isfile(filepath))
    )

all_merged_features_filename = "all_merged_features.csv"
all_merged_features_filepath = features_folderpath + all_merged_features_filename
