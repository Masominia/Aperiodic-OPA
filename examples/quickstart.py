"""Design a 32-emitter OPA for +/-60 deg steering and compare it with a uniform array and random layouts."""

import numpy as np

from aopa import Evaluator, Slit, Spec, random_layout, search, uniform_layout

spec = Spec(name="quickstart", n=32, gap_min=2.0, gap_max=5.0, aperture=103.0, steer_deg=60.0, pattern=Slit(0.5))
ev = Evaluator(spec)

uniform = ev.worst_db(uniform_layout(spec))
rng = np.random.default_rng(0)
random_best = min(ev.worst_db(random_layout(spec, rng)) for _ in range(200))
result = search(spec, seconds=60)

print(f"uniform 3.3 um pitch:      {uniform:6.2f} dB  (grating lobes)")
print(f"best of 200 random:        {random_best:6.2f} dB")
print(f"minimax layout, 1 minute:  {result.worst_db:6.2f} dB")
