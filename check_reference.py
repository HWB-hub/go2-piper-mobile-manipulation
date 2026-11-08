"""Check stand/turn gates, C2 foot-reference boundaries and reward degeneracies."""
import json
import torch
from pathlib import Path
from gait_reference import references, equivalent_speed

torch.set_default_dtype(torch.float64)
v=torch.tensor([[0.,0.,.48]])
assert (equivalent_speed(v)>.1).all()
assert not (v[:,:2].norm(dim=-1)+.15*v[:,2].abs()>.1).any()  # Former turn gate was inactive.
duty=(.60-.07*equivalent_speed(v)).item()
errors=[]
for boundary in (0.,duty):
    p=torch.tensor([boundary-1e-5,boundary,boundary+1e-5])
    c,h,xy,m=references(p,torch.full((3,),1.8),v.repeat(3,1))
    all_pos=torch.cat((xy,h[:,:,None]),dim=-1)
    jump=(all_pos[2]-all_pos[0]).abs().max().item()
    derivative_mismatch=((all_pos[2]-all_pos[1])-(all_pos[1]-all_pos[0])).abs().max().item()/1e-5
    assert jump<1e-4 and derivative_mismatch<1e-3
    assert (c[2]-c[0]).abs().max()<.001
    errors.append({'boundary':boundary,'position_difference_m':jump,'derivative_mismatch':derivative_mismatch})
p=torch.linspace(0.,1.,2001)[:-1]
c,h,xy,m=references(p,torch.full_like(p,1.8),v.repeat(len(p),1))
assert h.min()>=0 and h.max()>.045
assert torch.allclose(h[:,0],h[:,3]) and torch.allclose(h[:,1],h[:,2])
assert ((h[:,0]>.02)&(h[:,1]>.02)).sum()==0
assert xy[:,0,1].std()>.005 and xy[:,2,1].std()>.005
stand=references(p,torch.full_like(p,1.8),torch.zeros(len(p),3))
assert torch.all(stand[0]==1) and torch.all(stand[1]==0)
report={'passed':True,'pure_rotation_activates_all_legs':True,'continuous_boundaries':errors,
        'swing_height_peak_m':h.max().item(),'reference_cost_for_shuffling':(h/.04).square().mean().item()}
Path('reference_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
