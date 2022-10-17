FROM python:alpine3.16

RUN mkdir /discord_bot
WORKDIR /discord_bot
COPY main.py stable_diffusion_node_pb2.py stable_diffusion_node_pb2_grpc.py ./
COPY requirements_server.txt ./
RUN pip install -r requirements_server.txt

ENTRYPOINT python main.py