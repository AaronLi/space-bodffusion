from io import BytesIO

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import grpc

import stable_diffusion_node_pb2
from stable_diffusion_node_pb2_grpc import StableDiffusionNodeStub

if __name__ == '__main__':
    channel = grpc.insecure_channel('localhost:50051')
    stub = StableDiffusionNodeStub(channel)

    model_id = "CompVis/stable-diffusion-v1-4"
    device='cuda'

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16', use_auth_token=True).to(device)
    print('ready')
    while True:
        request = stub.GetTask(stable_diffusion_node_pb2.GetTaskRequest())
        print(f'received request: {request}')
        prompt = request.prompt
        width = request.width or 512
        height = request.height or 512
        num_inference_steps = request.num_inference_steps or 20
        negative_prompt = request.negative_prompt
        num_images_per_prompt = request.num_images_per_prompt or 1
        guidance_scale = request.guidance_scale or 7.5
        request_id = request.request_id
        print(
            f'''Executing
        prompt: {prompt},
        width: {width},
        height: {height},
        num_inference_steps: {num_inference_steps},
        neg_prompt: {negative_prompt},
        num_images_per_prompt: {num_images_per_prompt},
        guidance_scale: {guidance_scale},
        req_id: {request_id}
        ''')
        with autocast(device, dtype=torch.bfloat16):
            try:
                result = pipe(prompt, guidance_scale=guidance_scale, width=width, height=height, num_inference_steps=num_inference_steps, negative_prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt)
            except RuntimeError:
                print(request_id, prompt)
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