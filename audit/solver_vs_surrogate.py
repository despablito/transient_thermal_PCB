# Usage: python audit/solver_vs_surrogate.py . [N]   (run from repo root; first arg = repo path)
import sys, time, numpy as np, torch, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, sys.argv[1])
from scipy.integrate import solve_ivp
from data_gen import ThermalProperties, TransientThermalSolver, PowerProfileGenerator
from model import create_model
from train import TrainingConfig
torch.set_num_threads(1)
props = ThermalProperties(); solver = TransientThermalSolver(props); cfg = TrainingConfig()
model = create_model(verify_params=False)
model.load_state_dict(torch.load(sys.argv[1]+'/weights/best_model.pth', map_location='cpu', weights_only=False)['model_state_dict']); model.eval()
n=1024; ny=nx=32; dt=props.dt
def A_of(h):
    z=lambda t: np.zeros((32,32))
    return np.array([solver.heat_equation_rhs(0,np.eye(1,n,i).ravel(),z,h,0.0) for i in range(n)]).T  # dT/dt = A T (+ Q/rhoCp + h*Tamb/(th*rhoCp))
rng = np.random.default_rng(30000); N=int(sys.argv[2]); R=[]
for i in range(N):
    h = rng.uniform(*props.h_range); Tamb = rng.uniform(*props.T_amb_range)
    g = PowerProfileGenerator(props, seed=30000+i); P,_ = g.generate_power_sequence(100); Q = g.create_power_interpolator(P, dt)
    sols={}
    for name,ms in (('shipped',np.inf),('fixed',dt/4)):
        s = solve_ivp(lambda t,T: solver.heat_equation_rhs(t,T,Q,h,Tamb),(0,10),np.full(n,Tamb),method='BDF',t_eval=np.linspace(0,10,101),rtol=1e-3,atol=1e-4,max_step=ms)
        sols[name]=s.y.T.reshape(-1,32,32)[1:]
    A = A_of(h); src = lambda q: (q.ravel()/props.heat_capacity_volumetric) + (h*Tamb/props.thickness/props.heat_capacity_volumetric)
    # edge handling identical to FDM rhs (rows copied) -> A already encodes it (rows copy neighbour rhs); constant term too:
    def c_vec(q):
        z = solver.heat_equation_rhs(0, np.zeros(n), lambda t: q, h, Tamb); return z
    lu = spla.splu(sp.csc_matrix(np.eye(n)-dt*A))
    T = np.full(n,Tamb); ie=[]; Pp=np.zeros((32,32))
    t0=time.perf_counter()
    for k in range(100):
        T = lu.solve(T + dt*c_vec(0.5*(Pp+P[k]))); ie.append(T.reshape(32,32)); Pp=P[k]
    t_ie=(time.perf_counter()-t0)/100; ie=np.array(ie)
    hm=torch.full((1,1,32,32),h)/cfg.h_scale; tm=torch.full((1,1,32,32),Tamb)/cfg.T_scale; Tp=torch.full((1,1,32,32),Tamb)/cfg.T_scale; Pq=torch.zeros((1,1,32,32)); nn=[]
    t0=time.perf_counter()
    with torch.no_grad():
        for k in range(100):
            Pt=torch.from_numpy(P[k].astype(np.float32)).view(1,1,32,32)/cfg.P_scale; Tn=Tp+model(Pt,Pq,Tp,tm,hm); nn.append(Tn.numpy()[0,0]*cfg.T_scale); Tp,Pq=Tn,Pt
    t_nn=(time.perf_counter()-t0)/100; nn=np.array(nn)
    gt=sols['fixed']; pk=lambda X: X.reshape(100,-1).max(1)
    r=dict(corrupt=float(np.abs(sols['shipped']-gt).max()), mae_nn_vs_shipped=np.abs(nn-sols['shipped']).mean(), mae_nn=np.abs(nn-gt).mean(), mae_ie=np.abs(ie-gt).mean(), pk_nn=np.abs(pk(nn)-pk(gt)).mean(), pk_ie=np.abs(pk(ie)-pk(gt)).mean(), t_nn_ms=t_nn*1e3, t_ie_ms=t_ie*1e3)
    R.append(r); print(i, {k:round(float(v),2) for k,v in r.items()}, flush=True)
bad=[r for r in R if r['corrupt']>5]
print('corrupted (>5C) sequences:', len(bad), '/', N)
for k in R[0]: print(k, 'mean', round(float(np.mean([r[k] for r in R])),3), 'median', round(float(np.median([r[k] for r in R])),3))
