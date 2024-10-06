![Title](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral_london_hackathon/readme_img/img1.png?raw=true)

# Inspiration

Today’s robots are like computers from the 80s - very complicated and only usable by experts with years of experience. Programming them to do even one simple task is tough and requires a lot of technical knowledge. For robots to become everyday helpers, whether for repetitive tasks or heavy object lifting, learning new skills needs to be much easier.

Most of us learn new skills by watching YouTube tutorials. The question is, can robots do the same - learn a skill just by watching a human in a video?

# Introduction

![Keyframes](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral_london_hackathon/readme_img/img2.png?raw=true)

![Summary](https://github.com/johanndiep/language-models-trajectory-generators/blob/main_mistral_london_hackathon/readme_img/img3.png?raw=true)

Le CopyChat revolutionizes how robots learn by enabling them to mimic tasks demonstrated by humans through simple video footages. By extracting keyframes and summarizing actions using Pixtral, Le CopyChat translates these into commands that robots can follow. This approach simplifies robot programming, making it accessible for everyday tasks. Developed during the Mistral AI London Hackathon, the project showcases a pipeline that allows robots to learn new skills from videos, much like humans do.

## Branch Information
- **main** - Original project with the OpenAI model.
- **main_mistral_london_hackathon** - New model integration from Mistral and video processing, learning, and execution capabilities.
- **samarth_mergers** - Adds additaional features in image segmentation and server-client communication for model processing.

## Differences from the Original Project

1. **New Model Integration**: We have integrated a new LLM model from Mistral.
2. **Video Processing**: Added video processing capabilities to the LLM model to allow capabilities of leaning from video without any intervention.
4. **Additional Scripts**: New scripts have been added in video summerizer aimed at processing video files with image segmentation and Mithral model.
4. **API Enhancements**: Added new API endpoints to support the Mistral model.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**:
    ```sh
    git clone -b main_mistral_london_hackathon https://github.com/johanndiep/language-models-trajectory-generators.git
    cd language-models-trajectory-generators
    ```

2. **Initialize and Update Submodules**:
    ```sh
    git submodule init
    git submodule update
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set Up Configuration**:
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
### Running the Server

To start the server for remote processing:

```sh
python server.py # at the Server
```

for the client side, change the name of models_server_client.py to models.py. Rest eveything will remain the same thoughout the process. Please note that the client code converts the model into a file and then sends it over to the server. For this reason your computer might seem to hang up for a while. Do not worry, it is just the big size of the model being managed.