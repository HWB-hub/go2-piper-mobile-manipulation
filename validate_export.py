"""Check exported policy against the actor saved in its training checkpoint."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as functional

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
args = parser.parse_args()
folder = Path(args.model_dir)
state = torch.load(folder/"model_final.pt",map_location="cpu",weights_only=False)["actor_state_dict"]
with (folder/"policy_jit.pt").open("rb") as stream:
    policy = torch.jit.load(stream,map_location="cpu")
torch.manual_seed(19871)
inputs = torch.randn(256,state["mlp.0.weight"].shape[1])
manual = inputs.clone()
with torch.inference_mode():
    for index in (0,2,4,6):
        manual = functional.linear(manual,state[f"mlp.{index}.weight"],state[f"mlp.{index}.bias"])
        if index!=6:
            manual = functional.elu(manual)
    exported = policy(inputs)
error = (exported-manual).abs().max().item()
assert exported.shape==(256,18) and exported.isfinite().all() and error<1e-6
result = {"input_dimension":inputs.shape[1],"output_dimension":18,"random_cases":256,
          "max_absolute_difference":error,"all_outputs_finite":True,"passed":True}
(folder/"export_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
