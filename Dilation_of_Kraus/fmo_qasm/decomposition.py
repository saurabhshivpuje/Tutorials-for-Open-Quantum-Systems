import numpy as np
from numpy.linalg import inv

def is_unitary(A, tol=1e-5):
    # checks whether matrix A is unitary, within optional tolerance
    return np.allclose(np.matmul(A, A.conj().T), np.eye(A.shape[0], dtype=complex), rtol=tol)

def top_row_col(A, tol=1e-5):
    # checks whether the top row and column of matrix A are a 1 followed by 0's
    # used for moving down to a smaller submatrix in decomposition
    arr = np.zeros(A.shape[0], dtype=complex)
    arr[0] = 1
    return np.allclose(A[0:,0], arr, rtol=tol) and np.allclose(A[0,0:], arr, rtol=tol)

def next_nonzero(A, tol=1e-5):
    # returns an integer representing the index of the next non-zero off-diagonal
    # entry in the first column
    # returns -1 if all entries are 0
    for i in range(1,A.shape[1]):
        if not np.isclose(A[i,0], 0, rtol=tol):
            return i
    return -1

def add_pad(A, pad):
    # adds pad rows and columns with 1 on the diagonal and 0's on the off-diagonals
    # to the beginning of matrix A
    B = np.eye(pad+A.shape[0], dtype=complex)
    B[pad:,pad:] = A
    return B

def decompose(A, tol=1e-5):
    #return a list of arrays [U1, U2, ...] such that ...U2U1A = I
    assert is_unitary(A, tol)
    assert A.shape[0] == A.shape[1]

    # sub-fuction that does the decomposition recursively
    # A is the (current) matrix we are trying to decompose
    # pad is the number of levels we have already decomposed
    # lst is a list of nxn two-level unitaries that give the decomposition of
    # the initial matrix A
    def rec_decompose(A, pad, lst):
        if A.shape[0] <= 2:
            return lst
        # if the top row and columns are 1 followed by 0's, we can move down to
        # a smaller submatrix of A
        if top_row_col(A, tol):
            # if the current matrix is 3x3, we just need to change a 2x2 submatrix
            # to make A unitary, then we're done
            if A.shape == (3,3):
                # creates the final two-level unitary
                Un = np.eye(A.shape[0], dtype=complex)
                Un[1,1] = A[1,1].conj()
                Un[1,2] = A[2,1].conj()
                Un[2,1] = A[1,2].conj()
                Un[2,2] = A[2,2].conj()

                lst.append(add_pad(Un, pad))
                return lst
            else:
                # otherwise, we move to a smaller submatrix and increment padding
                return rec_decompose(A[1:,1:], pad+1, lst)
        # determine the next non-zero index
        # this will help us construct the appropriate two-level unitary
        nn = next_nonzero(A, tol)
        # if n == -1, then only the diagonal term may be non-zero
        # in this case, the construction of the two-level unitary is different
        if nn == -1:
            Un = np.eye(A.shape[0], dtype=complex)
            Un[0,0] = A[0,0].conj()
            lst.append(add_pad(Un, pad))

            return rec_decompose(np.matmul(Un,A), pad, lst)
        else:
            a = A[0,0]
            b = A[nn,0]
            mag = np.sqrt(a*a.conj() + b*b.conj())

            # Un is the two-level unitary
            # constructed in accordance with the method in Nielsen&Chuang
            Un = np.eye(A.shape[0], dtype=complex)
            Un[0,0] = a.conj() / mag
            Un[0,nn] = b.conj() / mag
            Un[nn,0] = b / mag
            Un[nn,nn] = -a / mag

            # add Un to the list of two-level unitaries
            lst.append(add_pad(Un, pad))
            # recursive call with updated A = Un*A
            return rec_decompose(np.matmul(Un,A), pad, lst)

    return rec_decompose(A, 0, [])

# single-qubit gate decomposition
def single_qubit_U(U, tol=1e-5):
    # returns alpha, beta, delta, gamma that are used to construct an arbitrary
    # 2x2 unitary matrix as a sequence of single-qubit gates
    # see the function U_from_abgd for how these variables are used
    assert U.shape == (2,2)
    assert is_unitary(U, tol)


    if np.isclose(np.abs(U[0,0]), 1, rtol=tol):
        # sometimes np.abs(U[0,0]) is ever so slightly greater than 1 causing problems with np.arccos
        gamma = 0
    else:
        gamma = 2*np.arccos(np.abs(U[0,0]))

    thetas = np.angle(U)

    coeffs = np.array([[1, -1/2, -1/2], [1, 1/2, -1/2], [1, 1/2, 1/2]])

    abd = np.matmul(inv(coeffs), np.array([thetas[0,0], thetas[1,0], thetas[1,1]]).T)

    return abd[0], abd[1], abd[2], gamma

def U_from_abgd(a,b,d,g):
    # recreates the 2x2 unitary matrix U from alpha, beta, delta, gamma
    return np.array([[np.exp(1j*(a-b/2-d/2))*np.cos(g/2), -np.exp(1j*(a-b/2+d/2))*np.sin(g/2)],
                    [np.exp(1j*(a+b/2-d/2))*np.sin(g/2), np.exp(1j*(a+b/2+d/2))*np.cos(g/2)]])

def construct_CU(U, circ, q, control=0, target=1, tol=1e-5, remove_id=True):
    # creates the gate sequence for a singly-controlled operation U
    # control: index of the control qubit
    # target: index of the target qubit (U is applied to this qubit if the control
    # qubit is 1)
    # remove_id: if this is set to True, gates which evaluate to identity will
    # not be applied
    assert is_unitary(U, tol)
    assert U.shape == (2,2)
    assert control != target

    # TODO: more single-qubit gate detection?
    # this checks if the desired controlled operation is the CNOT gate,
    # saving some time
    if np.allclose(U, np.array([[0,1],[1,0]],dtype=complex)):
        circ.cx(q[control], q[target])
        return circ

    a, b, d, g = single_qubit_U(U, tol)

    # C
    if not remove_id or (d-b)/2 != 0:
        circ.rz((d-b)/2, q[target])

    circ.cx(q[control], q[target])

    # B
    if not remove_id or (d+b)/2 != 0:
        circ.rz(-(d+b)/2, q[target])
    if not remove_id or g/2 != 0:
        circ.ry(-g/2, q[target])

    circ.cx(q[control], q[target])

    # A
    if not remove_id or g/2 != 0:
        circ.ry(g/2, q[target])
    if not remove_id or b != 0:
        circ.rz(b, q[target])

    if not remove_id or a != 0:
        a=a*np.pi
        circ.u(0,0,a, q[control]) #changed gate u1 to p as u1 is deprecated

    return circ


def check_diagonal(U, tol=1e-5):
    # checks whether U is a diagonal matrix (and returns the non-1 indices along the diagonal if applicable)
    ds = []
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if i != j and not np.isclose(U[i,j], 0, rtol=tol):
                return False
            elif i == j and not np.isclose(U[i,j], 1, rtol=tol):
                ds.append(i)
    if len(ds) == 1:
        ds.append((ds[0]+1)%U.shape[0])
    ds.sort()
    return ds


def find_two_levels(U, tol=1e-5):
    # return the nondiagonal index i,j where U[i,j] != 0
    assert U.shape[0] == U.shape[1]

    d = U.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j and not np.isclose(U[i,j], 0, rtol=tol):
                break
        if i != j and not np.isclose(U[i,j], 0, rtol=tol):
                break

    template = np.eye(d, dtype=complex)
    template[i,i] = U[i,i]
    template[i,j] = U[i,j]
    template[j,i] = U[j,i]
    template[j,j] = U[j,j]
    assert np.allclose(template, U, rtol=tol)
    return [i,j]

def U_from_levels(U2, levels):
    # returns single-qubit unitary operation given levels
    U = np.zeros((2,2), dtype=complex)
    levels.sort()
    U[0,0] = U2[levels[0], levels[0]]
    U[0,1] = U2[levels[0], levels[1]]
    U[1,0] = U2[levels[1], levels[0]]
    U[1,1] = U2[levels[1], levels[1]]
    return U

# old function, please ignore!
def implement_U(U, circ, q, tol=1e-5):
    # constructs the gate sequence of U (4x4)
    assert U.shape == (4,4)

    decomp = decompose(U)
    decomp.reverse()

    for u in decomp:
        Un = u.conj().T

        if np.allclose(np.eye(4), Un, rtol=tol):
            continue

        cd = check_diagonal(Un, tol)
        if cd:
            lvls = cd
        else:
            lvls = find_two_levels(Un)
            lvls.sort()

        q1_U = U_from_levels(Un, lvls)

        if lvls == [0,1]:
            circ.x(q[1])
            construct_CU(q1_U, circ, q, control=1, target=0)
            circ.x(q[1])
        elif lvls == [0,2]:
            circ.x(q[0])
            construct_CU(q1_U, circ, q, control=0, target=1)
            circ.x(q[0])
        elif lvls == [0,3]:
            circ.cx(q[1], q[0])
            circ.x(q[0])
            construct_CU(q1_U, circ, q, control=0, target=1)
            circ.x(q[0])
            circ.cx(q[1], q[0])
        elif lvls == [1,2]:
            circ.cx(q[1], q[0])
            construct_CU(q1_U, circ, q, control=0, target=1)
            circ.cx(q[1], q[0])
        elif lvls == [1,3]:
            construct_CU(q1_U, circ, q, control=0, target=1)
        elif lvls == [2,3]:
            construct_CU(q1_U, circ, q, control=1, target=0)

    return circ

# the following algorithms (Flip, AndTemp, And, Conditional) are from Rieffel&Polak
# I used them early for implementing a multiply-controlled gate
# please ignore now!
def Flip(a, b, circ, q):
    # a and b are lists denoting qubits in QuantumRegister q of QuantumCircuit circ
    assert len(a) == len(b) + 1

    if len(a) == 2:
        circ.ccx(q[int(a[1])], q[int(a[0])], q[int(b[0])])
    else:
        circ.ccx(q[int(a[-1])], q[int(b[-2])], q[int(b[-1])])
        Flip(a[:-1], b[:-1], circ, q)
        circ.ccx(q[int(a[-1])], q[int(b[-2])], q[int(b[-1])])
    return

def AndTemp(a, b, c, circ, q):
    assert len(b) == 1
    if len(a) == 2:
        circ.ccx(q[int(a[1])], q[int(a[0])], q[int(b[0])])
    else:
        Flip(a, b+c, circ, q)
        Flip(a[:-1], c)
    return

def And(a, b, t, circ, q):
    # t is a temporary qubit
    assert len(b) == 1
    assert len(t) == 1

    if len(a) == 1:
        circ.cx(q[int(a[0])], q[int(b[0])])
    elif len(a) == 2:
        circ.ccx(q[int(a[1])], q[int(a[0])], q[int(b[0])])
    else:
        m = len(a)
        k = int(np.floor(m/2))
        if m%2 == 0:
            j = k-2
        else:
            j = k-1
        AndTemp(a[k:], t, a[0:j+1], circ, q)
        AndTemp(a[0:j+1]+t, b, a[k:k+j-1], circ, q)
        AndTemp(a[k:], t, a[0:j+1], circ, q)
    return

def Conditional(z, Q, a, b, t, circ, q):
    # basically a fancy C^N(U)
    # t: 2 temporary qubits
    assert len(t) == 2
    assert len(z) == len(a)

    m = len(z)

    for i in range(m):
        if z[i] == 0:
            circ.x(q[int(a[i])])

    And(a, t[0:1], t[1:2], circ, q)
    construct_CU(Q, circ, q, control=t[0], target=b[0])
    And(a, t[0:1], t[1:2], circ, q)

    for i in range(m):
        if z[i] == 0:
            circ.x(q[int(a[i])])

    return

def CNU(z, U, a, b, t, circ, q, tol=1e-5):
    # N-controlled U operation on register b if z and a match
    # z: list of 1s and 0s that determine which qubits require X gate before the
    # multipley-controlled operation
    # U: the unitary matrix which is applied to the target qubit
    # a: list of indices of the control qubits
    # b: one-element list of the target qubit
    # t: list of indices of the work qubits
    assert is_unitary(U, tol)
    assert len(z) == len(a)
    assert len(a) == len(t) + 1

    m = len(z)

    for i in range(m):
        if z[i] == 0:
            circ.x(q[int(a[i])])

    if m == 1:
        construct_CU(U, circ, q, control=int(a[0]), target=int(b[0]))
        if z[0] == 0:
            circ.x(q[int(a[0])])
        return circ

    circ.ccx(q[int(a[0])], q[int(a[1])], q[int(t[0])])

    for i in range(2,m):
        circ.ccx(q[int(a[i])], q[int(t[i-2])], q[int(t[i-1])])

    construct_CU(U, circ, q, control=int(t[-1]), target=int(b[0]))

    for i in range(m-1,1,-1):
        circ.ccx(q[int(a[i])], q[int(t[i-2])], q[int(t[i-1])])

    circ.ccx(q[int(a[0])], q[int(a[1])], q[int(t[0])])

    for i in range(m):
        if z[i] == 0:
            circ.x(q[int(a[i])])

    return circ


# This is the main function! This creates the circuit for any unitary matrix!
def general_implement(U, circ, q, tol=1e-5):
    # U: the unitary matrix to implement
    # circ: the QuantumCircuit object to which the gates are applied
    # q: the QuantumRegister
    # tol: optional parameter for specifying the level of accuracy of many of the
    # matrix operations
    assert is_unitary(U)
    assert U.shape[0] == U.shape[1]

    # enlarge U if necessary
    d = U.shape[0]
    if not np.log2(d).is_integer():
        n = int(np.ceil(np.log2(d)))
        # pad = 2**n - d
        new_U = np.eye(2**n, dtype=complex)
        # new_U[pad:,pad:] = U
        new_U[:d,:d] = U
        U = new_U


    # X gate defined here for C^N-NOT
    X = np.array([[0,1],[1,0]], dtype=complex)


    # dimension of the matrix
    d = U.shape[0]
    # number of qubits
    n = int(np.log2(d))

    # decomposition of U into two-level unitaries
    decomp = decompose(U)
    decomp.reverse()

    for u in decomp:
        Un = u.conj().T

        if np.allclose(np.eye(d, dtype=complex), Un, rtol=tol):
            # ignore any unit matrices produced by decompose
            continue

        cd = check_diagonal(Un, tol)
        if cd:
            lvls = cd
        else:
            lvls = find_two_levels(Un)
            lvls.sort()

        q1_U = U_from_levels(Un, lvls)

        # gray code
        lvls_bin = ((np.array(lvls)[:,None] & (1 << np.arange(n))) > 0).astype(int)
        lvls_targets = ((np.array([d-2,d-1])[:,None] & (1 << np.arange(n))) > 0).astype(int)

        for i in range(1,-1,-1):
            for j in range(n):
                if lvls_bin[i,j] != lvls_targets[i,j]:
                    z = np.delete(lvls_bin[i],j)
                    a = np.delete(np.arange(n,dtype=int),j)
                    b = [j]
                    t = np.arange(n,n+len(a)-1,dtype=int)
                    # Conditional(z, X, a, b, t, circ, q)
                    CNU(z, X, a, b, t, circ, q)
                    lvls_bin[i,j] = lvls_targets[i,j]

        # controlled q1_U
        # Conditional(np.ones(n-1), q1_U, np.arange(1,n,dtype=int), [0], [n,n+1], circ, q)
        z = np.ones(n-1)
        a = np.arange(1,n,dtype=int)
        b = [0]
        t = np.arange(n,n+len(a)-1,dtype=int)
        CNU(z, q1_U, a, b, t, circ, q)

        # uncompute gray code
        lvls_bin = ((np.array(lvls)[:,None] & (1 << np.arange(n))) > 0).astype(int)

        for i in range(2):
            for j in range(n-1,-1,-1):
                if lvls_targets[i,j] != lvls_bin[i,j]:
                    z = np.delete(lvls_targets[i],j)
                    a = np.delete(np.arange(n,dtype=int),j)
                    b = [j]
                    t = np.arange(n,n+len(a)-1,dtype=int)
                    # Conditional(z, X, a, b, t, circ, q)
                    CNU(z, X, a, b, t, circ, q)
                    lvls_targets[i,j] = lvls_bin[i,j]

        circ.barrier()


    return circ
