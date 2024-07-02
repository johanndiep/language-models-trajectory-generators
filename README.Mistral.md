## Introduction to Le Chat Robot

![Title](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral/assets/title.png)

Controlling robotic arms requires a profound grasp of concepts like object detection and arm trajectory planning, restricting their use to specialized engineers and researchers. To democratize access to these advanced machines (e.g. for retail workers or farmers), it is important to enable human-like control. 

Thus, the project aims to fine-tune the Mistral LLM via its API to generate low-level commands from user commands and camera inputs, directly applicable to the robot actuators. Conducted solely in simulation, this project seeks to bridge the accessibility gap for advanced robotic technology.

This work is based on the publication ["Language Models as Zero-Shot Trajectory Generators"](https://arxiv.org/abs/2310.11604). 

## Starting the Simulator with Mistral

This repository is a fork of the [original repository by Teyun Kwon](https://github.com/kwonathan/language-models-trajectory-generators). Setting up the environment and running the script is almost the same as in the original (please refer to [README.md](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral/README.md) file for more information on how to set up the requirements and file structure), but two more parameters are required to run the Mistral fine-tuned model:
- for `--language_model` you need to specify the mistral fine-tuning model ID (ex: "ft:open-mistral-7b:daf5e488:20240430:c1bed559"). The code has been modified to detect whether the inputted language model is a GPT or a Mistral model and executes the needed functions.
- a new flag is added `--token` that passes the Mistral API key to the function that calls the fine-tuning API. This argument is also required, otherwise the system throws a permission error.

The python command overall looks like:

```
python main.py --robot franka --token <mistral_api_key> --language_model <ft:open-mistral-7b...>
```
The other available arguments from the original repo still work, if required by the user. Once the system and the models have been loaded, the simulator starts, and the user will have to input the prompt to tell the robot what to do:
```
Enter a command:
```

## Fine-tuning Improvements

### Dataset Generation
The dataset was provided to us by the first author of the original repo and paper ([Teyun Kwon](https://www.linkedin.com/in/john-teyun-kwon/)). The dataset consisted of 130 examples from testing the original codebase, utilising GPT models as the LLMs. Each entry of the dataset was a raw terminal output from the simulator run, similar to what a user would get if they run the system themselves and collect the terminal outputs. In the dataset, each entry was a run with a different command; there were a total of 35 unique commands, each repeated several times. Examples of similar commands can be found in the `/outputs` folder in the original repo. 

The raw data had to be cleaned and formatted into the standard for Mistral datasets to allow for fine-tuning. The structure used is the same as the default instruct data structure outlined in https://docs.mistral.ai/capabilities/finetuning/#1-default-instruct. The script `create_finetune_dataset.py` was written to process the data into that structure. The inputs are the raw dataset folder path `--data_path` and the name of the terminal output text files `--data_file_name` (the file for each run was named the same, only the sub-folder paths differed). The script would then find all dataset files with that name and:
1. Do pre- and post-cleaning of the string, removing terminal output artifacts, irrelevant to the data.
2. Find the `user`, `assistant` and `system` speakers in the data and extract their messages/prompts.
3. Organise the data in the right format for the Mistral fine-tuning, such as ensuring only one "system" prompt is present.

The end result was two JSONL files, train (124 example conversations) and validate (6 example conversations), that were used to fine-tune a mistral-7B model.

### Fine-tuning Results

Before the fine-tuning step, the simulation was executed with a base Mistral-7B model (`open-mistral-7b`) to test its performance. The model was reluctant to follow well the prompts, especially not adhering to the prompts to stop generating code after some point.

To combat this, prompt engineering was used to make the model produce outputs more aligned with the expectations for the task. The prompt snippet that worked best is shown below. It was included in the `prompts/main_prompt.py` and was part of all the testing of all models (fine-tuned and non-finetuned).
```
*** Important for you to follow ***
Stop generation after each python code block to wait for it to finish executing before continuing with your plan.
Explicitly say "Stopping Generation" after the generated code block and then stop!
```

The original `language-models-trajectory-generators` repository code was changed to allow for the LLM in the system to be swapped to a user-specified Mistral fine-tuning model (see section `Starting the Simulator with Mistral` above). The main function for this is `get_mistral_output` in `models.py`. The function uses `requests` to call the Mistral API, as this was found to be easier to work with because the input messages are in the right format (list of dicts) for the curl commands, but would need to be amended to `ChatMessage` for the Python SDK.

The fine-tuning was done in accordance with the instructions on creating a fine-tuning job from the Mistral website (https://docs.mistral.ai/capabilities/finetuning/#create-a-fine-tuning-job). Some results are presented below:

#### Base Mistral-7B

The non-finetuned model was not able to perform the commands specified to it, even after the aforementioned prompt engineering. Rarely (~10% of the time), the model would be able to execute a trajectory, but that trajectory would be very wrong (see `finetune_results/pick_up_bottle.mp4` for a video example of this). 

![Failure to pick up the bottle](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral/nonfinetune_results/pick_up_bottle.gif)

This indicates that the model's understanding of the task/command is poor. More often, the model would get stuck outputting wrong function calls, not recognising the errors that it has made in the previous output and thus not correcting the new outputs. This also applied when Mistral's specialized coding model (`codestral-latest`) was used instead. An example below (the pasted raw terminal output from a run) shows how the model is unable to progress with the simulation, as it is not able to produce no-error trajectory plans.

##### Failed Example:

Command: `move box left`

Terminal output (terminal artifacts removed for ease of reading):
```
...
user:
Running code block 1 of your previous response resulted in the following error:
Traceback (most recent call last):
  File "$HOME/language-models-trajectory-generators/main.py", line 131, in <module>
    exec(code)
  File "<string>", line 3, in <module>
  File "$HOME/language-models-trajectory-generators/api.py", line 142, in task_completed
    masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
  File "$HOME/language-models-trajectory-generators/models.py", line 119, in get_xmem_output
    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
  File "$HOME/language-models-trajectory-generators/venv_robot/lib/python3.10/site-packages/PIL/Image.py", line 3131, in open
    fp = builtins.open(filename, "rb")
IsADirectoryError: [Errno 21] Is a directory: '.'

Can you output a modified code block to resolve this error?

assistant: Let's set the `xmem_input_path` variable to the current working directory.

python
# Set the xmem input path to the current working directory
config.xmem_input_path = "."

user:

assistant: Now that the end-effector has moved back up to clear the tabletop, the task has now been completed. Let's call the `task_completed` function.

python
# Task completed
task_completed()
STOPPING GENERATION

user:
Running code block 1 of your previous response resulted in the following error:
Traceback (most recent call last):
  File "$HOME/language-models-trajectory-generators/main.py", line 131, in <module>
    exec(code)
  File "<string>", line 3, in <module>
  File "$HOME/language-models-trajectory-generators/api.py", line 142, in task_completed
    masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
  File "$HOME/language-models-trajectory-generators/models.py", line 119, in get_xmem_output
    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
  File "$HOME/language-models-trajectory-generators/venv_robot/lib/python3.10/site-packages/PIL/Image.py", line 3131, in open
    fp = builtins.open(filename, "rb")
IsADirectoryError: [Errno 21] Is a directory: '.'

Can you output a modified code block to resolve this error?

assistant: Let's set the `xmem_input_path` variable to the current working directory.

python
# Set the xmem input path to the current working directory
config.xmem_input_path = "."

user:

assistant: Now that the end-effector has moved back up to clear the tabletop, the task has now been completed. Let's call the `task_completed` function.

python
# Task completed
task_completed()
STOPPING GENERATION

user:
Running code block 1 of your previous response resulted in the following error:
Traceback (most recent call last):
  File "$HOME/language-models-trajectory-generators/main.py", line 131, in <module>
    exec(code)
  File "<string>", line 3, in <module>
  File "$HOME/language-models-trajectory-generators/api.py", line 142, in task_completed
    masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
  File "$HOME/language-models-trajectory-generators/models.py", line 119, in get_xmem_output
    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
  File "$HOME/language-models-trajectory-generators/venv_robot/lib/python3.10/site-packages/PIL/Image.py", line 3131, in open
    fp = builtins.open(filename, "rb")
IsADirectoryError: [Errno 21] Is a directory: '.'

Can you output a modified code block to resolve this error?

assistant: Let's set the `xmem_input_path` variable to the current working directory.

python
# Set the xmem input path to the current working directory
config.xmem_input_path = "."

user:

assistant: Now that the end-effector has moved back up to clear the tabletop, the task has now been completed. Let's call the `task_completed` function.

python
# Task completed
user:
Running code block 1 of your previous response resulted in the following error:
Traceback (most recent call last):
  File "$HOME/language-models-trajectory-generators/main.py", line 131, in <module>
    exec(code)
  File "<string>", line 3, in <module>
  File "$HOME/language-models-trajectory-generators/api.py", line 142, in task_completed
    masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
  File "$HOME/language-models-trajectory-generators/models.py", line 119, in get_xmem_output
    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
  File "$HOME/language-models-trajectory-generators/venv_robot/lib/python3.10/site-packages/PIL/Image.py", line 3131, in open
    fp = builtins.open(filename, "rb")
IsADirectoryError: [Errno 21] Is a directory: '.'

Can you output a modified code block to resolve this error?

```

#### Fine-tuned Mistral-7B

The finetuning training was done on 5 epochs, to test the model performance, compared to the base, non-finetuned Mistral-7B model. The fine-tuned model performed much better than the base model. It was able to successfully finish some of the given tasks, with the task of picking up the bottle being successful ~60% of the time. It was able to generate trajectories and execute the functions given in the system prompt much more consistently. The model's actions showed that it was able to understand the task, and tried to execute it. Some successful runs can be found in `finetune_results/`, labelled with "_success".

![Successfully picking up the bottle](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral/finetune_results/pick_up_bottle_success_1.gif)

Sometimes the model tried to fulfill the task, but was not successful, yet close. As examples, in `finetune_results/knock_over_bottle.mp4` it can be seen that the model tried to horizontally knock the mustard bottle over, but it stopped right before the bottle due to poor trajectory planning. In `finetune_results/move_box_left_fail_2.mp4` the robot tried to pick up the box, then move it left, just as instructed, and place it down; the only issue with the run was that it missed the box initially. These failures show that the model has gained understanding from the fine-tuning on how to control the robot and plan trajectories. The base Mistral-7B model produced many more errors and was not able to plan and execute almost any trajectories, and even it did, they were incoherent.

![Failure to move the box to the left](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral/finetune_results/move_box_left_fail_2.gif)


#### Performance

To measure the performance of the base model and the fine-tuned model, we tested with 3 different commands, to test the ability of the model to deal with singular objects. The reasoning was that if the model is able to handle single objects, the testing would move to multiple objects. Each command was tested 5 times on each model, for consistency, with the `bottle` and `box` items referring to `mustard bottle` and `cracker box`. The results are presented in the table below:


<center>Table: Model success rate (in %) on user commands

|  | Base Mistral-7B  | Fine-tuned Mistral-7B |
|:----|:----------------:|:------------------:|
| move box left      | 0%   | 0%  |
| pick up bottle     | 0%   | 60%   |
| knock over bottle  | 0%   | 20%   |

</center>

The fine-tuned model has shown great improvement on all tasks. Even on the `move box left` task, that seeminly has stayed at 0/6, the fine-tuning model performs much better, as it was able to generate 3-4 times reasonable trajectories, but was erroneous at some steps, and missed the target. The results that the fine-tuned Mistral-7B model shows are even more impressive, given:
- the small fine-tuning data provided to the model (130 examples)
- the low fine-tune training time (5 epochs)
- the size of the model (the model is 7B parameters, much smaller than all the GPT models that were used for the original `language-models-trajectory-generators` repo)

## Conclusion

The work presented is a testament that small LLMs can be trained on complex tasks, such as controlling robot motion, and with few examples to learn from, drastically improve their performance. Further work is required to build on top of this, increasing the data size and training, but this beginning shows how promising LLMs are for robot control, making the task of robot control much simpler for users. 

Small LLMs can be put on edge devises and robots directly, eliminating the latency of devices talking to servers, and greatly decreasing costs for running large, computationally heavy, slow at inference LLMs.

## Authors
- [Stefan Karmakov](https://www.linkedin.com/in/stefankarmakov/) (karmakovst@gmail.com) 
- [Johann Diep](https://www.linkedin.com/in/johann-diep/) (johanndiep@gmail.com)

Special thanks go to [Teyun Kwon](https://www.linkedin.com/in/john-teyun-kwon/), the author of the original repository and the paper ["Language Models as Zero-Shot Trajectory Generators"](https://arxiv.org/abs/2310.11604), for providing the dataset and sharing his insights.


