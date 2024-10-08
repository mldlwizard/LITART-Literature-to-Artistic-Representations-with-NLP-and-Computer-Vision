from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

prompt = '''Golbasto Momarem Evlame Gurdilo Shefin Mully Ully Gue, most mighty Emperor of Lilliput, delight and terror of the universe, proposes to the Man-Mountain Gulliver that he make the following oath: 1) The man-mountain Gulliver shall not depart from his dominions without our license under our great seal.'''

model_id = "runwayml/stable-diffusion-v1-5"
runway_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
runway_pipe = runway_pipe.to("cuda")

for i in range(5):
    runway_pipe(prompt).images[0].save(f"results/runawayml{i}.png")

# ______________________________________________________________________________


model_id = "stabilityai/stable-diffusion-2-1"
stability_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
stability_pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
stability_pipe = stability_pipe.to("cuda")

for i in range(5):
    stability_pipe(prompt).images[0].save(f"results/stabilityAI{i}.png")
