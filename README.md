# TinyGPT
A very small language model based on the transformer architecture

# Environment

The models are trained and run within a Docker Container. The image can be modified by changing `/dockerfile` (for example, to install additional dependencies) and its configuration for running can be adjusted within `/compose.yaml`. The container has accessed to the files in `/src` via a volume mounted on `/home/tiny-gpt/src`.

To setup the Docker Container, run the following commands in a terminal window

1. Build the docker image

    `cd this_repo`

    `docker compose build`

2. Start the docker container

    `docker compose start`
