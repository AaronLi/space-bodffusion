import argparse
import functools
import queue
import random
import time
import uuid
import threading
from io import BytesIO

import grpc
import pypresence
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from wakepy import keepawake

import stable_diffusion_node_pb2
from stable_diffusion_node_pb2_grpc import StableDiffusionNodeStub


def get_task(retries = 10) -> stable_diffusion_node_pb2.StableDiffusionRequest:
    for i in range(retries):
        try:
            return stub.GetTask(stable_diffusion_node_pb2.GetTaskRequest())
        except grpc.RpcError as e:
            print('failed to get task: ', e)
        except Exception as e:
            print('unknown error', e)
        time.sleep(random.random() * 2**i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Space Bodffusion processing node')
    parser.add_argument('server', type=str, help='server address')
    args = parser.parse_args()
    channel = grpc.insecure_channel(args.server)
    stub = StableDiffusionNodeStub(channel)

    model_id = "CompVis/stable-diffusion-v1-4"
    device='cuda'

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16', use_auth_token=True).to(device)

    presence = pypresence.Presence('942678157281079358')
    presence.connect()
    start_time = int(time.time())
    print('ready')
    with keepawake():
        while True:
            presence.update(start=start_time)
            request = get_task()
            if request is None:
                continue
            if not request.valid:
                continue
            prompt = request.prompt
            width = request.width or 512
            height = request.height or 512
            total_steps = request.num_inference_steps or 20
            negative_prompt = request.negative_prompt
            num_images_per_prompt = request.num_images_per_prompt or 1
            guidance_scale = request.guidance_scale or 7.5
            request_id = request.request_id
            print(f'received request:')
            print(
                f'''Executing
            prompt: {prompt},
            width: {width},
            height: {height},
            num_inference_steps: {total_steps},
            neg_prompt: {negative_prompt},
            num_images_per_prompt: {num_images_per_prompt},
            guidance_scale: {guidance_scale},
            req_id: {uuid.UUID(bytes=request_id)}
            ''')
            with autocast(device, dtype=torch.bfloat16):
                try:
                    updates_queue = queue.Queue()
                    def updates_callback(step: int, timestep: int, latents: torch.FloatTensor):
                        updates_queue.put(stable_diffusion_node_pb2.UpdateProgressMessage(request_id=request_id, current_step = step, num_steps=total_steps))

                    def send_update():
                        while True:
                            val = updates_queue.get(True)
                            if val is None:
                                break
                            yield val

                    update_thread = threading.Thread(target=lambda: stub.UpdateProgress(send_update()))
                    update_thread.start()
                    result = pipe(prompt, guidance_scale=guidance_scale, width=width, height=height, num_inference_steps=total_steps, negative_prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt, callback=updates_callback, callback_steps=total_steps//10)
                    updates_queue.put(None)
                    update_thread.join()
                except RuntimeError:
                    stub.PostResult(stable_diffusion_node_pb2.StableDiffusionResponse(request_id= request_id, images = [], prompt=prompt, out_of_memory = True))
                    continue
            images = result[0]
            nsfw_detected = result[1]
            out_images = []
            for image in images:
                bytes_out = BytesIO()
                image.save(bytes_out, 'jpeg')
                out_images.append(bytes_out.getvalue())
            stub.PostResult(stable_diffusion_node_pb2.StableDiffusionResponse(prompt=prompt, request_id = request_id, images=out_images))
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print('completed request')