# Usage: python audit/count_skipped_power.py . [N]   (run from repo root; first arg = repo path)
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from scipy.integrate import solve_ivp
from data_gen import ThermalProperties, TransientThermalSolver, PowerProfileGenerator
props = ThermalProperties(); solver = TransientThermalSolver(props)
rng = np.random.default_rng(20000)
bad=0; tot=0
for i in range(int(sys.argv[2]) if len(sys.argv)>2 else 60):
    h = rng.uniform(*props.h_range); Tamb = rng.uniform(*props.T_amb_range)
    g = PowerProfileGenerator(props, seed=20000+i); P,_ = g.generate_power_sequence(100)
    Q = g.create_power_interpolator(P, props.dt)
    T0 = np.full((32,32), Tamb)
    sol = solve_ivp(lambda t,T: solver.heat_equation_rhs(t,T,Q,h,Tamb), (0,10), T0.ravel(), method='BDF', t_eval=np.linspace(0,10,101), rtol=1e-3, atol=1e-4)
    Tm = sol.y.T.reshape(-1,32,32)
    flat = np.ptp(Tm) < 1e-6
    nz = (P.reshape(100,-1).max(1)>0)
    tot+=1; bad+=flat
    if flat or i<3: print(i, 'flat' if flat else 'ok', 'status',sol.status,'nfev',sol.nfev,'frames with power',nz.sum(),'first power frame',np.argmax(nz) if nz.any() else None, 'nsteps', len(sol.t))
print('flat sequences', bad, '/', tot)
