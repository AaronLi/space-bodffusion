import asyncio
import io
import math
import os
import sys
import uuid
from collections import namedtuple

import grpc
import discord
from discord import Option, ApplicationContext, utils, User, TextChannel

import stable_diffusion_node_pb2
import stable_diffusion_node_pb2_grpc
from concurrent import futures
from stable_diffusion_node_pb2_grpc import StableDiffusionNodeServicer

CommandQueueEntry = namedtuple('CommandQueueEntry', ('request', 'initial_response_message', 'channel', 'user'))

class StableDiffusionNodeServer(StableDiffusionNodeServicer):
    async def UpdateProgress(self, request, context):
        progress_percentage = request.current_step/request.num_steps

        task = self.task_data[request.request_id]

        default_text = get_base_response_message(task.user.display_name, task.request.num_images_per_prompt,
                                                 task.request.prompt, task.request.width, task.request.height)
        green_boxes = math.ceil(20*progress_percentage)
        progress_bar = f"{green_boxes*'🟩'}{(20 - green_boxes) * '🟥'}"
        await task.initial_response_message.edit_original_response(content=default_text+f"\nIn progress: {progress_bar} {int(100*progress_percentage)}%")

        return stable_diffusion_node_pb2.UpdateProgressResponse()

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
            task_info.request.request_id = uuid.uuid4().bytes

            # put old task data into new entry
            self.task_data[task_info.request.request_id] = task_info

            default_text = get_base_response_message(task_info.user.display_name, task_info.request.num_images_per_prompt,
                                                     task_info.request.prompt, task_info.request.width, task_info.request.height)

            # update message to say timed out
            # requeue task
            await asyncio.gather(
                self.task_data[task_info.request.request_id].initial_response_message.edit_original_response(content=default_text+"\nTimed Out.\nPrompt will be retried"),
                self.request_queue.put(task_info)
            )

    def add_timeout(self, task: stable_diffusion_node_pb2.StableDiffusionRequest):
        self.timeouts[task.request_id] = asyncio.create_task(self.create_timeout(task))

    def add_task_data(self, task_data: CommandQueueEntry):
        self.task_data[task_data.request.request_id] = task_data

    async def GetTask(self, request, context):
        print('worker awaiting task')
        try:
            # get task to execute
            task = await asyncio.wait_for(self.request_queue.get(), 10)

            # create timeout and put in timeouts dict
            self.add_timeout(task.request)

            default_text = get_base_response_message(task.user.display_name, task.request.num_images_per_prompt, task.request.prompt, task.request.width, task.request.height)
            # notify progress started
            await task.initial_response_message.edit_original_response(content=default_text+"\nWorker started working on task...")

            self.add_task_data(task)
            return task.request
        except asyncio.exceptions.TimeoutError:
            # timed out waiting for task
            return stable_diffusion_node_pb2.StableDiffusionRequest(valid=False)

    async def PostResult(self, request, context):
        try:
            request_info = self.task_data.pop(request.request_id)

            default_text = get_base_response_message(request_info.user.display_name, request_info.request.num_images_per_prompt,
                                                     request_info.request.prompt, request_info.request.width, request_info.request.height)
            # request didn't time out, cancel timeout watcher
            del self.timeouts[request.request_id]
            if request.out_of_memory:
                await request_info.initial_response_message.edit_original_response(
                    content=default_text + f'\n{request_info.user.mention} Worker ran out of memory trying to fulfill your request (try reducing number of images or resolution)'
                )
            else:
                await request_info.initial_response_message.edit_original_response(
                    content=default_text+f'\nFinished request for {request_info.user.mention}\n**{request.prompt}**',
                    files=[discord.File(io.BytesIO(image), filename=f'{i}.jpg') for i, image in enumerate(request.images)]
                )
        except KeyError:
            # key does not exist in data store, message came from timed out request
            pass
        return stable_diffusion_node_pb2.PostResultResponse()

    def __init__(self, command_queue: asyncio.Queue[CommandQueueEntry]) -> None:
        self.request_queue = command_queue
        self.task_data: dict[bytes, CommandQueueEntry] = {}
        self.timeouts = {}


def get_base_response_message(display_name, num_images, prompt, width, height) -> str:
    return f'{display_name} submitted prompt for {num_images}x\"{utils.escape_markdown(prompt)}\" with resolution {width}x{height}'

if __name__ == '__main__':
    discord_bot = discord.Bot()
    command_queue = asyncio.Queue(maxsize=128)

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
        response = get_base_response_message(ctx.user.display_name, num_images, prompt, width, height)
        print(response)

        if command_queue.full():
            await ctx.send_response("Queue is full, please wait")
            return

        request_uuid = uuid.uuid4().bytes
        initial_message = await ctx.send_response(content=response)
        await command_queue.put(
            CommandQueueEntry(
                request=stable_diffusion_node_pb2.StableDiffusionRequest(
            prompt=prompt,
            request_id=request_uuid,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            valid=True
                ),
                initial_response_message = initial_message,
                channel=ctx.channel,
                user=ctx.user
            )
        )

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    stable_diffusion_node_pb2_grpc.add_StableDiffusionNodeServicer_to_server(StableDiffusionNodeServer(command_queue), server)

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
