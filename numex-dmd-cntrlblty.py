# ## Dependencies
# scipy (tested with version 1.5.2)
# numpy (tested with version 1.19.5)
# matplotlib (tested with version 3.3.2)

# ## Howto
# Run
# `python3 numex-dmd-cntrlblty.py`


import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt


epsi = 5e-1  # Problem parameter: default: 5e-1
tE = 10  # end time: default: 10
Nts = 100  # number of time points and snapshots: default 100
N = 5  # dimension of the model will be 2*N: default 5

tmesh = np.linspace(0, tE, Nts+1)  # the time grid
inival = np.arange(2*N)+1


def comp_dmda(solmat=None):
    xmat = solmat[:, :-1]
    zmat = solmat[:, 1:]
    dmdA = zmat.dot(np.linalg.pinv(xmat))

    return dmdA


def plotit(data, timerange=None, fignum=100, figsize=(5, 3), titlestr=''):

    plt.figure(fignum, figsize=figsize)
    N = data.shape[1]
    plt.rcParams["axes.prop_cycle"] = \
        plt.cycler("color", plt.cm.plasma(np.linspace(0.1, .9, N)))
    plt.plot(timerange, data, linewidth=2)
    plt.xlabel('time $t$')
    plt.title(titlestr)
    plt.xlim((min(timerange), max(timerange)))


# the coefficient matrix
scalevec = np.arange(N)
scaledeye = np.diag(scalevec)
nzero = np.zeros((N, N))
A = np.vstack([np.hstack([nzero, 1./epsi*scaledeye]),
               np.hstack([nzero, -epsi*scaledeye])])


def expat(t):
    # we compute the exact transfer matrix exp(At)
    phit = np.exp(-epsi*scalevec*t)
    psit = 1./epsi**2*(1-phit)
    expat = np.vstack([np.hstack([np.eye(N), np.diag(psit)]),
                       np.hstack([nzero, np.diag(phit)])])
    return expat


exasoltrajeclist = [expat(t).dot(inival) for t in tmesh]
exasoltrajec = np.array(exasoltrajeclist)
soltrajec = exasoltrajec

dmdA = comp_dmda(solmat=soltrajec.T)

# ## CHAP: identifying the reachable space
Cspacegenlist = [inival]
for kkk in range(2*N-1):
    Cspacegenlist.append(A.dot(Cspacegenlist[-1]))
Cspacegenmat = np.array(Cspacegenlist).T

ucsx, scsx, vcsx = spla.svd(Cspacegenmat)
keepscindices = (scsx/scsx[0]) > 1e-12
nkcs = keepscindices.sum()
bcspc = ucsx[:, :nkcs]
bucspc = ucsx[:, nkcs:]
print('dim of `C(A,x_0)`: {0}'.format(nkcs))

# ## Initial Value from CSpace
cspccoeffs = np.ones((nkcs, 1))
cspinival = (bcspc @ cspccoeffs).flatten()
dmdsol = [cspinival]
for k in np.arange(Nts):
    dmdsol.append(dmdA.dot(dmdsol[-1]))
dmdsol = np.array(dmdsol)
cspxsol = np.array([expat(t).dot(cspinival) for t in tmesh])
plotit(cspxsol, timerange=tmesh, fignum=101,
       titlestr='$v_0\\in  C_{\\Delta, x_0}$: Exact Solution')
plotit(dmdsol, timerange=tmesh, fignum=201,
       titlestr='$v_0\\in  C_{\\Delta, x_0}$: DMD Solution')

# ## Initial Value outside CSpace
ucspinival = (bucspc @ np.ones((nkcs, 1))).flatten()
dmdsol = [ucspinival]
for k in np.arange(Nts):
    dmdsol.append(dmdA.dot(dmdsol[-1]))
dmdsol = np.array(dmdsol)
cspxsol = np.array([expat(t).dot(ucspinival) for t in tmesh])
plotit(cspxsol, timerange=tmesh, fignum=102,
       titlestr='$v_0\\in C_{\\Delta, x_0}^\\perp $: Exact Solution')
plotit(dmdsol, timerange=tmesh, fignum=202,
       titlestr='$v_0\\in C_{\\Delta, x_0}^\\perp $: DMD Solution')

# ## CHAP: data transformation
tmat = np.eye(2*N) + np.diag(np.ones((2*N-1,)), 1)
tmatinv = np.linalg.inv(tmat)

bdmdA = comp_dmda(solmat=tmat@soltrajec.T)

bdmdsol = [tmat@cspinival]
for k in np.arange(Nts):
    bdmdsol.append(bdmdA.dot(bdmdsol[-1]))
bdmdsol = np.array(bdmdsol)

plotit(bdmdsol@tmatinv.T, timerange=tmesh, fignum=301,
       titlestr='$v_0\\in C_{\\Delta, x_0}$: T-DMD Solution')

bdmdsol = [tmat@ucspinival]
for k in np.arange(Nts):
    bdmdsol.append(bdmdA.dot(bdmdsol[-1]))
bdmdsol = np.array(bdmdsol)

plotit(bdmdsol@tmatinv.T, timerange=tmesh, fignum=302,
       titlestr='$v_0\\in  C_{\\Delta, x_0}^\\perp $: T-DMD Solution')

plt.show()
