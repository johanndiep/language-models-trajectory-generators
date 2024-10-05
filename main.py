import numpy as np
import math
import openai
import torch
import os
import sys
import argparse
import traceback
import multiprocessing
import logging
import functools
import models
import config
import pybullet as p
import pybullet_data
import socket
from time import sleep
from multiprocessing import Process, Pipe
from contextlib import redirect_stdout
from io import StringIO
from lang_sam import LangSAM
from api import API
from env import run_simulation_environment
from prompts.main_prompt import MAIN_PROMPT
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from config import OK, PROGRESS, FAIL, ENDC

sys.path.append(os.path.join(os.path.dirname(__file__), 'XMem'))
sys.path.append("./XMem/")

print = functools.partial(print, flush=True)

from XMem.model.network import XMem


def pybullet_server(robot, port=6000):
    # Set up the socket server to listen for incoming connections
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(1)
    print(f"PyBullet server started on port {port}")

    # Accept incoming client connections
    client_socket, addr = server_socket.accept()
    print(f"Connection established with client at {addr}")

    try:
        # Initialize PyBullet in DIRECT mode (headless)
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Dynamically select the robot model based on the provided argument
        if robot == "sawyer":
            model_path = "sawyer_robot/sawyer.urdf"  # Update with the correct path for Sawyer
            robot_start_position = [0, 0, 1]
            robot_start_orientation = [0, 0, 0, 1]
        elif robot == "franka":
            model_path = "franka_panda/panda.urdf"  # Update with the correct path for Franka
            robot_start_position = [0, 0, 1]
            robot_start_orientation = [0, 0, 0, 1]
        else:
            raise ValueError(f"Unknown robot type: {robot}")

        # Load the selected robot model and a plane for context
        planeId = p.loadURDF("plane.urdf")
        robotId = p.loadURDF(model_path, robot_start_position, robot_start_orientation)

        # Collect model information to send to the client
        models_info = {
            "plane": {"path": "plane.urdf", "position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            "robot": {"path": model_path, "position": robot_start_position, "orientation": robot_start_orientation}
        }

        # Send model information to the client before starting the simulation
        client_socket.sendall(str(models_info).encode())
        print(f"Model data sent to client: {models_info}")

        # Run the simulation loop and send data to the client
        while True:
            # Step the simulation
            p.stepSimulation()
            # Get the position and orientation of the robot
            pos, orn = p.getBasePositionAndOrientation(robotId)
            # Send the robot's position and orientation as a formatted string to the client
            message = f"{pos[0]},{pos[1]},{pos[2]},{orn[0]},{orn[1]},{orn[2]},{orn[3]}\n"

            try:
                client_socket.sendall(message.encode())  # Send data to the client
            except (BrokenPipeError, ConnectionResetError):
                print(f"Client at {addr} disconnected. Closing connection.")
                break  # Exit the loop if the client disconnects unexpectedly

            # Sleep to maintain the desired simulation rate
            sleep(0.01)

    except Exception as e:
        print(f"Error in server: {e}")

    finally:
        # Clean up and close sockets
        p.disconnect()
        client_socket.close()
        server_socket.close()
        print("Server shut down gracefully.")



if __name__ == "__main__":

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Parse args
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", choices=["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], default="gpt-4", help="select language model")
    parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer", help="select robot")
    parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default", help="select mode to run")
    parser.add_argument("-p", "--port", type=int, default=6000, help="Port to run the server on")
    args = parser.parse_args()

    # Logging
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    # Device
    if torch.cuda.is_available():
        logger.info("Using GPU.")
        device = torch.device("cuda")
    else:
        logger.info("CUDA not available. Please connect to a GPU instance if possible.")
        device = torch.device("cpu")

    torch.set_grad_enabled(False)

    # Start the PyBullet server in a separate process with the chosen robot model
    server_process = multiprocessing.Process(target=pybullet_server, args=(args.robot, args.port))
    server_process.start()


    # Load models
    langsam_model = LangSAM()
    xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)

    # API set-up
    main_connection, env_connection = Pipe()
    api = API(args, main_connection, logger, langsam_model, xmem_model, device)

    detect_object = api.detect_object
    execute_trajectory = api.execute_trajectory
    open_gripper = api.open_gripper
    close_gripper = api.close_gripper
    task_completed = api.task_completed

    # Start environment process
    env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, env_connection, logger])
    env_process.start()

    [env_connection_message] = main_connection.recv()
    logger.info(env_connection_message)

    # User input
    command = input("Enter a command: ")
    api.command = command

    # ChatGPT
    logger.info(PROGRESS + "STARTING TASK..." + ENDC)

    messages = []

    error = False

    new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)

    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "system")
    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

    while True:

        while not api.completed_task:

            new_prompt = ""

            if len(messages[-1]["content"].split("```python")) > 1:

                code_block = messages[-1]["content"].split("```python")

                block_number = 0

                for block in code_block:
                    if len(block.split("```")) > 1:
                        code = block.split("```")[0]
                        block_number += 1
                        try:
                            f = StringIO()
                            with redirect_stdout(f):
                                exec(code)
                        except Exception:
                            error_message = traceback.format_exc()
                            new_prompt += ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                            new_prompt += "\n"
                            error = True
                        else:
                            s = f.getvalue()
                            error = False
                            if s != "" and len(s) < 2000:
                                new_prompt += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]", s)
                                new_prompt += "\n"
                                error = True

            if error:
                api.completed_task = False
                api.failed_task = False

            if not api.completed_task:

                if api.failed_task:

                    logger.info(FAIL + "FAILED TASK! Generating summary of the task execution attempt..." + ENDC)

                    new_prompt += TASK_SUMMARY_PROMPT
                    new_prompt += "\n"

                    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "user")
                    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

                    logger.info(PROGRESS + "RETRYING TASK..." + ENDC)

                    new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)
                    new_prompt += "\n"
                    new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages[-1]["content"])

                    messages = []

                    error = False

                    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "system")
                    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

                    api.failed_task = False

                else:

                    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                    messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "user")
                    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

        logger.info(OK + "FINISHED TASK!" + ENDC)

        new_prompt = input("Enter a command: ")

        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
        messages = models.get_chatgpt_output(args.language_model, new_prompt, messages, "user")
        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

        api.completed_task = False
