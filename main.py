import asyncio
import io
import os
import sys
import uuid
from typing import Dict, Tuple

import grpc
import discord
from discord import Option, ApplicationContext, utils, User, TextChannel

import stable_diffusion_node_pb2
import stable_diffusion_node_pb2_grpc
from concurrent import futures
from stable_diffusion_node_pb2_grpc import StableDiffusionNodeServicer

class StableDiffusionNodeServer(StableDiffusionNodeServicer):
    async def create_timeout(self, task, timeout=15):
        # if a request times out, remove any info about the original task and create a new request using the same task info but with a new request id
        await asyncio.sleep(timeout)
        if task.request_id in self.task_data:
            # remove self from timeouts
            del self.timeouts[task.request_id]

            # task was not finished
            print(f"task {uuid.UUID(bytes=task.request_id)}: \"{task.prompt}\" timed out. Reinserting into queue...")

            # get and remove old task data
            task_info = self.task_data.pop(task.request_id)

            #generate new request_id
            task.request_id = uuid.uuid4().bytes

            # put old task data into new entry
            self.task_data[task.request_id] = task_info

            #requeue task
            await self.request_queue.put(task)

    async def GetTask(self, request, context):
        print('worker awaiting task')
        try:
            # get task to execute
            task = await asyncio.wait_for(self.request_queue.get(), 10)

            # create timeout and put in timeouts dict
            self.timeouts[task.request_id] = asyncio.create_task(self.create_timeout(task))
            return task
        except asyncio.exceptions.TimeoutError:
            # timed out waiting for task
            return stable_diffusion_node_pb2.StableDiffusionRequest(valid=False)

    async def PostResult(self, request, context):
        try:
            request_info = self.task_data.pop(request.request_id)
            # request didn't time out, cancel timeout watcher
            del self.timeouts[request.request_id]
            if request.out_of_memory:
                await request_info[0].send(
                    content=f'{request_info[1].mention} Worker ran out of memory trying to fulfill your request (try reducing number of images or resolution)'
                )
            else:
                await request_info[0].send(
                    content=f'Finished request for {request_info[1].mention}\n**{request.prompt}**',
                    files=[discord.File(io.BytesIO(image), filename=f'{i}.jpg') for i, image in enumerate(request.images)]
                )
        except KeyError:
            # key does not exist in data store, message came from timed out request
            pass
        return stable_diffusion_node_pb2.PostResultResponse()

    def __init__(self, command_queue: asyncio.Queue, task_data: Dict[bytes, Tuple[TextChannel, User]]) -> None:
        self.request_queue = command_queue
        self.task_data = task_data
        self.timeouts = {}


if __name__ == '__main__':
    discord_bot = discord.Bot()
    command_queue = asyncio.Queue(maxsize=128)
    task_data = {}

    @discord_bot.slash_command()
    async def stable_diffusion(
            ctx: ApplicationContext,
            prompt: Option(str, desription='Stable Diffusion Prompt', required=True),
            num_images: Option(
                int,
                name='count',
                description='Number of images to generate',
                required=False,
                default=1,
                min_value=1,
                max_value=5),
            width: Option(
                int,
                required=False,
                default=512,
                min_value=16,
                max_value=768
            ),
            height: Option(
                int,
                required=False,
                default=512,
                min_value=16,
                max_value=768
            )
    ):
        response = f'{ctx.user.display_name} submitted prompt for {num_images}x\"{utils.escape_markdown(prompt)}\" with resolution {width}x{height}'
        print(response)

        if command_queue.full():
            await ctx.send_response("Queue is full, please wait")
            return

        request_uuid = uuid.uuid4().bytes
        task_data[request_uuid] = (ctx.channel, ctx.user)
        await ctx.send_response(content=response)
        await command_queue.put(stable_diffusion_node_pb2.StableDiffusionRequest(
            prompt=prompt,
            request_id=request_uuid,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            valid=True
        ))

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    stable_diffusion_node_pb2_grpc.add_StableDiffusionNodeServicer_to_server(StableDiffusionNodeServer(command_queue, task_data), server)

    server.add_insecure_port('0.0.0.0:50051')

    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if bot_token is None:
        print('No bot token specified (specify using DISCORD_BOT_TOKEN environment variable)')
        sys.exit(1)
    loop = asyncio.get_event_loop()
    loop.create_task(server.start(), name='grpc_server')
    loop.create_task(discord_bot.start(token=bot_token), name='discord_bot')
    print('ready')
    loop.run_forever()
