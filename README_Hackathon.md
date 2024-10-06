![Title](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral_london_hackathon/readme_img/img1.png?raw=true)

# Inspiration

Today’s robots are like computers from the 80s - very complicated and only usable by experts with years of experience. Programming them to do even one simple task is tough and requires a lot of technical knowledge. For robots to become everyday helpers, whether for repetitive tasks or heavy object lifting, learning new skills needs to be much easier.

Most of us learn new skills by watching YouTube tutorials. The question is, can robots do the same - learn a skill just by watching a human in a video?

# Introduction

Le CopyChat revolutionizes how robots learn by enabling them to mimic tasks demonstrated by humans through simple video footages. By extracting keyframes and summarizing actions using Pixtral, Le CopyChat translates these into commands that robots can follow. This approach simplifies robot programming, making it accessible for everyday tasks. Developed during the Mistral AI London Hackathon, the project showcases a pipeline that allows robots to learn new skills from videos, much like humans do.

For robots to become everyday helpers, whether for repetitive tasks or heavy lifting, we need to make teaching them as simple as watching a tutorial video.

In a [previous Mistral hackathon](youtube.com/watch?v=_vsRd8RsCKo), we showed that a fine-tuned Mistral LLM could generate code to control a robot just by understanding natural language robot commands, such as "build a can tower" or "put the can on a plate". For the current hackathon, we took it further by adding a video summarizer that watches a human demonstration, summarizes the actions in the video, and translates them into commands the robot can follow.

# Architecture

![Keyframes](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral_london_hackathon/readme_img/img2.png?raw=true)

First, we separate the human demonstration video into individual frames. To keep only the important frames where distinct human actions occur, we used a method based on visual embedding similarity to get the keyframes.

![Summary](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral_london_hackathon/readme_img/img3.png?raw=true)

After extracting the keyframes, we used Pixtral and Mistral Large to create a summary of the video. We started by generating an initial description based on the first frame using Pixtral. Then, with the same model we created descriptions for each pair of consecutive frames and the previous description, capturing the changes over time. Finally, we summarized all these descriptions into a cohesive summary using the Mistral Large model. Another Mistral Large model then converted this summary into a robot command.

# Branches within this Code Repository
- **main_mistral_london_hackathon**: Main runcode for this hackathon. 
- **sem_keyframe_and_server_accel**: Adds additional features in semantic keyframe extration and Nexus server acceleration for simulated object segmentation.
- **main_mistral**: Contains the simulation with Mistral models finetuned on custom manipulation dataset. 
- **main**: Original project by [Teyun Kwon](https://www.linkedin.com/in/john-teyun-kwon?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAADLSx-4Br89qB1k_S51afaIxMyepxtVURa0&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BZ61AYGs7TU23QrU%2FgZlpAA%3D%3D).

## Setup Instructions

### Prerequisites

This codebase has been successfully tested on Ubuntu 22.04.5 LTS.

### Installation

1. **Clone the repository**:
    ```sh
    git clone -b main_mistral_london_hackathon https://github.com/johanndiep/language-models-trajectory-generators.git
    cd language-models-trajectory-generators
    ```

2. **Initialize and update submodules**:
    ```sh
    git submodule init
    git submodule update
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Setup configuration**:
    - Update `config.py` with your API keys and endpoints for the Mistral model.
    ```
    mkdir -p images/trajectory
    mkdir XMem/saves
    wget -P XMem/saves https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
    ``` 

5. **Run the simulation**:
    ```sh
    python main.py --robot franka
    ```  

### Running the Nexus Server

To start the server for accelerated remote processing of the simulated object segmentation:

```sh
python server.py
```

For the client side, change the name of `models_server_client.py` to `models.py`. Rest will remain the same thoughout the process. Please note that the client code converts the model into a file and then sends it over to the server. For this reason your computer might seem to hang up for a while. Do not worry, it is just the big size of the model being managed.