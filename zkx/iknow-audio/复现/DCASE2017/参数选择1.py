# Code Contributor - Ankit Shah - ankit.tronix@gmail.com
# Adapted for Python 3 by user request
import pafy
import time
import datetime
import itertools
import os
import sys
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

# Format audio - 16 bit Signed PCM audio sampled at 44.1kHz
def format_audio(input_audio_file, output_audio_file):
    temp_audio_file = output_audio_file.split('.wav')[0] + '_temp.wav'
    cmdstring = "ffmpeg -loglevel panic -i %s -ac 1 -ar 44100 %s" % (input_audio_file, temp_audio_file)
    os.system(cmdstring)
    cmdstring1 = "sox %s -G -b 16 -r 44100 %s" % (temp_audio_file, output_audio_file)
    os.system(cmdstring1)
    cmdstring2 = "rm -rf %s" % (temp_audio_file)
    os.system(cmdstring2)

# Trim audio based on start time and duration of audio.
def trim_audio(input_audio_file, output_audio_file, start_time, duration):
    cmdstring = "sox %s %s trim %s %s" % (input_audio_file, output_audio_file, start_time, duration)
    os.system(cmdstring)

def multi_run_wrapper(args):
    return download_audio_method(*args)

# Method to download audio - Downloads the best audio available for audio id,
# calls the formatting audio function and then segments the audio formatted based on start and end time.
def download_audio_method(line, csv_file, output_root=None):
    query_id = line.split(",")[0]
    start_seconds = line.split(",")[1]
    end_seconds = line.split(",")[2]
    audio_duration = float(end_seconds) - float(start_seconds)
    # positive_labels = ','.join(line.split(",")[3:])
    print("Query -> " + query_id)
    # print("start_time -> " + start_seconds)
    # print("end_time -> " + end_seconds)
    # print("positive_labels -> " + positive_labels)
    url = "https://www.youtube.com/watch?v=" + query_id
    try:
        video = pafy.new(url)
        bestaudio = video.getbestaudio()

        # Determine output root directory
        if output_root is None:
            # Original behavior: generate folder names based on csv file path
            csv_base = os.path.splitext(os.path.basename(csv_file))[0]
            parent_dir = os.path.dirname(csv_file)
            folder_prefix = os.path.basename(parent_dir) if parent_dir else ""
            output_folder = os.path.join(os.getcwd(), f"{folder_prefix}_{csv_base}_audio_downloaded")
            formatted_folder = os.path.join(os.getcwd(), f"{folder_prefix}_{csv_base}_audio_formatted_downloaded")
            segmented_folder = os.path.join(os.getcwd(), f"{folder_prefix}_{csv_base}_audio_formatted_and_segmented_downloads")
        else:
            # Use user-specified root
            output_folder = os.path.join(output_root, "audio_downloaded")
            formatted_folder = os.path.join(output_root, "audio_formatted_downloaded")
            segmented_folder = os.path.join(output_root, "audio_formatted_and_segmented_downloads")

        # Create directories if they don't exist
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(formatted_folder, exist_ok=True)
        os.makedirs(segmented_folder, exist_ok=True)

        # Download best audio
        path_to_download = os.path.join(output_folder, "Y" + query_id + "." + bestaudio.extension)
        bestaudio.download(path_to_download)

        # Format to wav
        path_to_formatted_audio = os.path.join(formatted_folder, "Y" + query_id + ".wav")
        format_audio(path_to_download, path_to_formatted_audio)

        # Trim to segment
        path_to_segmented_audio = os.path.join(segmented_folder, "Y" + query_id + '_' + start_seconds + '_' + end_seconds + ".wav")
        trim_audio(path_to_formatted_audio, path_to_segmented_audio, start_seconds, audio_duration)

        # Remove intermediate folders (optional)
        # cmdstring2 = "rm -rf %s %s" % (output_folder, formatted_folder)
        # os.system(cmdstring2)
        # cmdstring3 = "rm -rf %s" % (formatted_folder)
        # os.system(cmdstring3)

        ex1 = ""
    except Exception as ex:
        ex1 = str(ex) + ',' + str(query_id)
        print("Error is ---> " + str(ex))
    return ex1

# Download audio - Reads 3 lines of input csv file at a time and passes them to multi_run_wrapper
# which calls download_audio_method to download the file based on id.
# Multiprocessing module spawns 3 process in parallel which runs download_audio_method.
def download_audio(csv_file, timestamp, output_root=None):
    error_log = 'error' + timestamp + '.log'
    with open(csv_file, "r") as segments_info_file:
        with open(error_log, "a") as fo:
            # We'll read lines in chunks of 3 manually to preserve original behavior
            lines = segments_info_file.readlines()
            # Process in chunks of 3
            for i in range(0, len(lines), 3):
                chunk = lines[i:i+3]
                lines_list = [(line.strip(), csv_file, output_root) for line in chunk]

                P = multiprocessing.Pool(3)
                exception = P.map(multi_run_wrapper, lines_list)
                for item in exception:
                    if item:
                        fo.write(str(item) + '\n')
                P.close()
                P.join()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python download_audio.py <csv_file> [output_root_directory]')
        print('Example: python download_audio.py groundtruth_strong_label_testing_set.csv /home/star/zkx/iknow-audio/data/DCASE_2017_evaluation_set_audio_files')
        sys.exit(1)

    csv_file = sys.argv[1]
    output_root = sys.argv[2] if len(sys.argv) == 3 else None

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    download_audio(csv_file, timestamp, output_root)