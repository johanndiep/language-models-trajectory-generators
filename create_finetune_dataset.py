import glob
import os
import json
from random import shuffle
import argparse


def clean_beginning(input_string):
    """
    The system prompt is always first, so remove any terminal artifacts that come before the prompt.
    Cleaning the data ensures better training examples, thus better model fine-tuning.
    """
    # Find the position of the first occurrence of "system:\n"
    position = input_string.find("system:\n")
    # If "system:\n" is found, remove all text before it
    if position != -1:
        return input_string[position:], True
    else:
        return "Beginning is wrong", False

def clean_ending(input_string):
    """
    Find the end of the useful part of the raw data and remove the terminal artifacts that follow
    after the last speaker has finished their prompt. Cleaning the data ensures better training 
    examples, thus better model fine-tuning.
    """
    lines = input_string.split('\n')
    speakers = ["user:", "assistant:", "system:"]
    info_substring = "[INFO/"

    last_speaker_index = -1
    info_index = -1

    # Iterate through lines to find the last speaker index
    for i, line in enumerate(lines):
        if any(line == speaker for speaker in speakers):
            last_speaker_index = i

    # Find the first occurrence of info_substring after the last speaker
    if last_speaker_index != -1:
        for i in range(last_speaker_index + 1, len(lines)):
            if info_substring in lines[i]:
                info_index = i
                break

    # If the info_substring is found after the last speaker, slice the lines accordingly
    if info_index != -1:
        cleaned_lines = lines[:info_index]
        cleaned_string = '\n'.join(cleaned_lines)
        return cleaned_string, True
    else:
        return "Ending is wrong", False
    
def check_if_not_failed(input_string):
    """ 
    Scan the document and check if in the run the system has failed.Remove failed instances 
    to allow cleaner dataset (empirically tested fails confuse Mistral-7B model more).
    """
    fail_message = "[INFO/MainProcess] FAILED TASK!"
    position = input_string.find(fail_message)
    # If the substring is found, keep everything before the substring and add 'stop'
    if position != -1:
        # if message is found -> task has failed
        return False
    else:
        return True

def extract_text_by_role(input_string):
    """ 
    Extract, in order, the occurences of the user, assistant and system speakers in the raw data
    and save them as dicts in a list, in accordance with Mistral dataset format
    """
    lines = input_string.split('\n')
    
    # Add ":" to speakers to remove false positives, after remove when writing to messages
    speakers = ["system:", "user:", "assistant:"]
    current_speaker = None
    messages = []
    system_count = 0

    # Iterate through each line
    for line in lines:
        # Check if the line is not a filler line
        if not line.startswith('[INFO/MainProcess]') and not line.startswith('[INFO/EnvProcess]'):
            # Check if the line indicates a new speaker
            if line in speakers:
                # For Mistral training purposes, "system" message can occur only once => 
                # add check and change to "user" for all subsequent "system" prompts
                if line == "system:":
                    system_count += 1
                    if system_count > 1:
                        line = "user:"
                current_speaker = line
                # Append a new dictionary for the new speaker
                messages.append({"role": current_speaker[:-1], "content": ""})
            elif current_speaker:
                # Append the current line to the last dictionary's value
                if messages[-1]["content"]:
                    messages[-1]["content"] += '\n' + line
                else:
                    messages[-1]["content"] += line

    # For Mistral training purposes, the conversation can't end with "user", so remove if it does
    if messages[-1]["role"] == "user":
        messages = messages[:-1]
    return {"messages": messages}

def save_to_jsonl(data, jsonl_name):
    """ Save dict {"messages": List[dicts]} to JSON Lines file
    """
    with open(f'{jsonl_name}.jsonl', 'w') as outfile:
        for message_dict in data:
            json.dump(message_dict, outfile)
            outfile.write('\n')

def main(args):
    folder_path = args.data_path
    pattern = os.path.join(folder_path, '**', args.data_file_name)
    # Use iglob to find all matching files recursively
    files = glob.iglob(pattern, recursive=True)

    messages_list = []
    for tso_path in files:

        with open(tso_path, 'r') as ep:
            data = ep.read()

        if check_if_not_failed(data):
            data_clean_start, success_bool = clean_beginning(data)
            if success_bool:
                data_clean_end, success_bool = clean_ending(data_clean_start)
                if success_bool:
                    messages_list.append(extract_text_by_role(data_clean_end))
                else:
                    print(f'{tso_path}:{data_clean_end}')
            else:
                print(f'{tso_path}:{data_clean_start}')
        else:
            print(tso_path)
    
    # Shuffle the list of message dicts
    shuffle(messages_list)
    data_size = len(messages_list)
    # Take 95% of messages for training, leave rest for validation
    train_cut = round(data_size*0.95)

    # Create train JSON Lines file
    data_train = messages_list[:train_cut]
    save_to_jsonl(data_train, 'fine_tuning_train_nofail')

    # Create validation JSON Lines file
    data_val = messages_list[train_cut:]
    save_to_jsonl(data_val, 'fine_tuning_val_nofail')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process folder of raw data into suitable Mistral Finetuning format")
    parser.add_argument("--data_path", "-p", type=str, required=True, help="Path to folder containing full dataset")
    # The data was organised in sub-folders and each data file had the same name
    parser.add_argument("--data_file_name", "-f", type=str, required=True, help="Name of the file in each dataset subfolder that contains raw data")
    args = parser.parse_args()

    main(args)