"""Linear-elastic Q4 plane-stress membrane solver for the in-plane part of SHELL181 mesh."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
try:
    from .material import MATERIALS, plane_stress_matrix
except ImportError:
    from material import MATERIALS, plane_stress_matrix

def shape_derivatives(xi,eta):
    return 0.25*np.array([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],[(1+eta),(1+xi)],[-(1+eta),(1-xi)]])

def b_matrix(xi,eta,xy):
    dN_nat=shape_derivatives(xi,eta)
    J=dN_nat.T@xy
    det=float(np.linalg.det(J))
    if det<=1e-12: raise ValueError(f'Non-positive Jacobian determinant: {det}')
    dN_xy=np.linalg.solve(J,dN_nat.T).T
    B=np.zeros((3,8))
    for a in range(4):
        B[0,2*a]=dN_xy[a,0]; B[1,2*a+1]=dN_xy[a,1]
        B[2,2*a]=dN_xy[a,1]; B[2,2*a+1]=dN_xy[a,0]
    return B,det

def assemble(nodes,elements,material_ids,thickness_by_section,section_ids):
    n=len(nodes); rows=[]; cols=[]; data=[]
    gp=1/np.sqrt(3); points=[(-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)]
    for e,q in enumerate(elements):
        mid=int(material_ids[e]); sid=int(section_ids[e])
        if mid not in MATERIALS: raise ValueError(f'No material definition for CDB material ID {mid}')
        if sid not in thickness_by_section: raise ValueError(f'No thickness for shell section {sid}')
        m=MATERIALS[mid]; D=plane_stress_matrix(m['E_MPa'],m['nu']); t=thickness_by_section[sid]
        xy=nodes[q]; Ke=np.zeros((8,8))
        for xi,eta in points:
            B,det=b_matrix(xi,eta,xy); Ke+=B.T@D@B*det*t
        dofs=np.array([[2*a,2*a+1] for a in q]).ravel()
        rr,cc=np.meshgrid(dofs,dofs,indexing='ij')
        rows.extend(rr.ravel()); cols.extend(cc.ravel()); data.extend(Ke.ravel())
    K=coo_matrix((data,(rows,cols)),shape=(2*n,2*n)).tocsr()
    return K

def build_tie_constraints(nodes, elements, material_ids):
    """Create exact translational ties between the three separately meshed bodies.

    Interface assumptions verified from this CDB geometry:
      * steel wall (MAT 3) to aluminium bracket (MAT 2): x = 0
      * aluminium bracket (MAT 2) to steel clamp (MAT 1): y = 8 mm
    The slave-side nodes are constrained to linear interpolation on the master edge.
    """
    n=len(nodes)
    node_sets={}
    for mid in (1,2,3):
        used=np.unique(elements[material_ids==mid].ravel())
        node_sets[mid]=used
    constraints=[]
    def add_edge_ties(master_nodes, slave_nodes, axis):
        master_nodes=np.asarray(master_nodes,dtype=int)
        slave_nodes=np.asarray(slave_nodes,dtype=int)
        if len(master_nodes)<2 or len(slave_nodes)==0:
            raise ValueError('Insufficient nodes on one of the bonded interfaces.')
        order=np.argsort(nodes[master_nodes,axis]); master_nodes=master_nodes[order]
        svals=nodes[slave_nodes,axis]
        mvals=nodes[master_nodes,axis]
        for sn,val in zip(slave_nodes,svals):
            if val < mvals[0]-1e-5 or val > mvals[-1]+1e-5:
                raise ValueError(f'Slave node {sn} falls outside master interface range.')
            j=int(np.searchsorted(mvals,val,side='right'))
            if j==0: pair=[(master_nodes[0],1.0)]
            elif j>=len(mvals): pair=[(master_nodes[-1],1.0)]
            else:
                a,b=master_nodes[j-1],master_nodes[j]
                xa,xb=nodes[a,axis],nodes[b,axis]
                if abs(xb-xa)<1e-12: pair=[(a,1.0)]
                else:
                    t=(val-xa)/(xb-xa); pair=[(a,1.0-t),(b,t)]
            for dof in (0,1):
                row={2*int(sn)+dof:1.0}
                for mn,w in pair: row[2*int(mn)+dof]=row.get(2*int(mn)+dof,0.0)-w
                constraints.append(row)
    wall_nodes=node_sets[3]; bracket_nodes=node_sets[2]; clamp_nodes=node_sets[1]
    wall_edge=wall_nodes[np.isclose(nodes[wall_nodes,0],0.0,atol=1e-5)]
    bracket_wall_edge=bracket_nodes[np.isclose(nodes[bracket_nodes,0],0.0,atol=1e-5)]
    bracket_edge=bracket_nodes[np.isclose(nodes[bracket_nodes,1],8.0,atol=1e-5)]
    clamp_edge=clamp_nodes[np.isclose(nodes[clamp_nodes,1],8.0,atol=1e-5)]
    add_edge_ties(wall_edge,bracket_wall_edge,axis=1)
    add_edge_ties(bracket_edge,clamp_edge,axis=0)
    rr=[]; cc=[]; dd=[]
    for r,row in enumerate(constraints):
        for dof,val in row.items(): rr.append(r);cc.append(dof);dd.append(val)
    from scipy.sparse import coo_matrix
    C=coo_matrix((dd,(rr,cc)),shape=(len(constraints),2*n)).tocsr()
    info={'wall_bracket_slave_nodes':int(len(bracket_wall_edge)),
          'bracket_clamp_slave_nodes':int(len(clamp_edge)),
          'tie_equations':int(C.shape[0]),
          'interfaces':['MAT3 wall to MAT2 bracket at x=0','MAT2 bracket to MAT1 clamp at y=8 mm']}
    return C,info

def solve(K,F,fixed_dofs,C=None):
    """Solve with zero-displacement supports and exact multipoint tie constraints."""
    from scipy.sparse import bmat, csr_matrix
    all_dofs=np.arange(K.shape[0]); free=np.setdiff1d(all_dofs,fixed_dofs)
    Kff=K[free][:,free].tocsr(); Ff=F[free]
    if C is not None and C.shape[0]:
        Cf=C[:,free].tocsr()
        # Constraints whose participating DOFs are all prescribed are already
        # satisfied by the supports; discard these zero rows before factorization.
        row_norm=np.asarray(abs(Cf).sum(axis=1)).ravel()
        keep=row_norm>1e-12
        Cf=Cf[keep]
        Ckeep=C[keep]
        A=bmat([[Kff,Cf.T],[Cf,None]],format='csc')
        rhs=np.concatenate([Ff,np.zeros(Cf.shape[0])])
        sol=spsolve(A,rhs)
        uf=sol[:len(free)]; lam=sol[len(free):]
        u=np.zeros(K.shape[0]);u[free]=uf
        reactions=K@u-F+Ckeep.T@lam if C.shape[0] and Cf.shape[0] else K@u-F
    else:
        u=np.zeros(K.shape[0]);u[free]=spsolve(Kff,Ff)
        reactions=K@u-F
    if not np.all(np.isfinite(u)): raise RuntimeError('Solver returned non-finite displacements.')
    return u,np.asarray(reactions).ravel()

def recover_stress(nodes,elements,material_ids,section_ids,thickness_by_section,u):
    gp=1/np.sqrt(3); points=[(-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)]
    stress=[]; strain=[]; vm=[]; centroids=[]
    for e,q in enumerate(elements):
        m=MATERIALS[int(material_ids[e])]; D=plane_stress_matrix(m['E_MPa'],m['nu'])
        ue=u[np.array([[2*a,2*a+1] for a in q]).ravel()]; xy=nodes[q]
        es=[]; ss=[]
        for xi,eta in points:
            B,_=b_matrix(xi,eta,xy); eps=B@ue; sig=D@eps
            es.append(eps); ss.append(sig)
        ss=np.asarray(ss); es=np.asarray(es)
        v=np.sqrt(ss[:,0]**2-ss[:,0]*ss[:,1]+ss[:,1]**2+3*ss[:,2]**2)
        stress.append(ss.mean(axis=0)); strain.append(es.mean(axis=0)); vm.append(v.max()); centroids.append(xy.mean(axis=0))
    return np.asarray(stress),np.asarray(strain),np.asarray(vm),np.asarray(centroids)


def recover_detailed_results(nodes, elements, material_ids, section_ids,
                             thickness_by_section, u):
    """Recover in-plane stress/strain fields and principal/equivalent measures.

    Equivalent elastic strain is calculated as the 3-D von-Mises equivalent
    strain using plane-stress epsilon_z = -nu/(1-nu)*(epsilon_x+epsilon_y).
    Principal stresses are the two in-plane principal values (sigma_1 >= sigma_2).
    All stresses are MPa; strains are dimensionless.
    """
    gp = 1.0 / np.sqrt(3.0)
    points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    rows = []
    for e, q in enumerate(elements):
        mat = MATERIALS[int(material_ids[e])]
        D = plane_stress_matrix(mat['E_MPa'], mat['nu'])
        ue = u[np.array([[2*a, 2*a+1] for a in q]).ravel()]
        xy = nodes[q]
        eps_gp, sig_gp = [], []
        for xi, eta in points:
            B, _ = b_matrix(xi, eta, xy)
            eps = B @ ue
            sig = D @ eps
            eps_gp.append(eps)
            sig_gp.append(sig)
        eps_gp = np.asarray(eps_gp)
        sig_gp = np.asarray(sig_gp)
        eps = eps_gp.mean(axis=0)
        sig = sig_gp.mean(axis=0)
        sx, sy, txy = sig
        mean = 0.5 * (sx + sy)
        radius = np.sqrt((0.5*(sx-sy))**2 + txy**2)
        p1, p2 = mean + radius, mean - radius
        vm_gp = np.sqrt(sig_gp[:,0]**2 - sig_gp[:,0]*sig_gp[:,1]
                        + sig_gp[:,1]**2 + 3.0*sig_gp[:,2]**2)
        nu = mat['nu']
        ez = -nu/(1.0-nu)*(eps[0]+eps[1])
        ex, ey, gamma_xy = eps
        emean = (ex + ey + ez)/3.0
        dx, dy, dz = ex-emean, ey-emean, ez-emean
        e_tensor_xy = 0.5*gamma_xy
        eq_strain = np.sqrt((2.0/3.0)*(dx*dx + dy*dy + dz*dz
                                      + 2.0*e_tensor_xy*e_tensor_xy))
        rows.append((eps[0], eps[1], gamma_xy, eq_strain,
                     sx, sy, txy, np.max(vm_gp), p1, p2, *xy.mean(axis=0)))
    a = np.asarray(rows)
    return {
        'strain': a[:,0:3], 'equivalent_elastic_strain': a[:,3],
        'stress': a[:,4:7], 'von_mises': a[:,7],
        'maximum_principal_stress': a[:,8],
        'minimum_principal_stress': a[:,9], 'centroids': a[:,10:12]
    }
