# Prompt: Conjugate Gradients Numba\n def
Synthesizing 10/10 solutions

=======

def CG(A, b, x, tol):
    r = b - A @ x
    p = r
    rsold = r.T @ r
    for i in range(len(b)):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol, maxiter):
    x = x0
    r = b - A(x)
    p = r
    rsold = r.dot(r)
    for i in range(maxiter):
        Ap = A(p)
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x, tol, maxiter):
    r = b - A(x)
    p = r
    rsold = r.T @ r

    for i in range(maxiter):
        Ap = A(p)
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            return x, i
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, maxiter

=======

def CG(x, b, A, tol, max_iter):
    r = b - A(x)
    p = r.copy()
    rsold = r.dot(r)
    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            return x, i
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, i

=======

def CG(A, b, x, tol=1e-6, max_iter=1000):
    r = b - A @ x
    p = r
    rsold = r.T @ r
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol=1e-6, max_iter=1000):
    x = x0
    r = b - A @ x
    p = r
    rsold = r @ r
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol=1e-5, maxiter=1000):
    r0 = b - A(x0)
    p0 = r0
    x = x0
    for i in range(maxiter):
        alpha = np.dot(r0, r0) / np.dot(p0, A(p0))
        x = x + alpha * p0
        r = r0 - alpha * A(p0)
        if np.sqrt(np.dot(r, r)) < tol:
            break
        beta = np.dot(r, r) / np.dot(r0, r0)
        p = r + beta * p0
        r0 = r
        p0 = p
    return x

=======

def CG(A, b, x0, tol=1e-6, maxIter=100):
    """
    Conjugate Gradients
    CG(A, b, x0, tol=1e-6, maxIter=100)
    A: Matrix of coefficients
    b: Vector of constants
    x0: Initial guess
    tol: Tolerance
    maxIter: Maximum iterations
    """
    r = b - A @ x0
    p = r
    x = x0
    for i in range(maxIter):
        a = np.dot(r, r) / np.dot(p, A @ p)
        x = x + a * p
        r = r - a * A @ p
        if np.linalg.norm(r) < tol:
            break
        b = np.dot(r, r) / np.dot(p, A @ p)
        p = r + b * p
    return x

=======

def CG_Numba(b, x, tol, maxIter, A):
    r = b - A.dot(x)
    p = r.copy()
    rsold = r.dot(r)
    for i in range(maxIter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            return x, i
        p = r + (rsnew/rsold) * p
        rsold = rsnew
    return x, maxIter

=======

def CG(x, A, b, max_iter=1000, tol=1e-5, verbose=False):
    """
    Conjugate Gradients Numba
    """
    r = b - A(x)
    p = r.copy()
    rsold = np.dot(r, r)
    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            if verbose:
                print('Conjugate Gradients Numba converged in', i, 'iterations')
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# Prompt: Conjugate Gradients cuPy\n def
Synthesizing 10/10 solutions

=======

def CG(A, b, x0, tol, maxiter):
    x = x0
    r = b - A @ x
    p = r
    rsold = r.T @ r
    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            return x, i
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, maxiter

=======

def CG(A, b, x, maxiter=100, tol=1e-5):
    r = b - A @ x
    p = r.copy()
    rsold = r.T @ r
    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.T @ r
        if rsnew < tol:
            print("Conjugate Gradients converged at iteration", i, "with residual", rsnew)
            return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    print("Conjugate Gradients did not converge after", maxiter, "iterations with residual", rsnew)
    return x

=======

def CG(A, b, x, tol=1e-10, maxiter=1000):
    r = b - A @ x
    p = r.copy()
    rsold = r.T @ r
    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol, maxiter):
    x = x0
    r = b - A @ x0
    p = r
    for i in range(maxiter):
        alpha = (r.T @ r) / (p.T @ A @ p)
        x = x + alpha * p
        r_new = r - alpha * A @ p
        if np.sqrt(r_new.T @ r_new) < tol:
            break
        beta = (r_new.T @ r_new) / (r.T @ r)
        p = r_new + beta * p
        r = r_new
    return x

=======

def CG(A, b, x, maxiter = 100, tol = 1e-8):
    r = b - A @ x
    p = r
    rsold = r.T @ r
    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x, maxiter=1000, tol=1e-6, verbose=False):
    r = b - A @ x
    p = r.copy()
    rsold = r.T @ r
    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.T @ r
        if rsnew < tol:
            if verbose:
                print('CG converged in {} iterations'.format(i))
            break
        p = r + rsnew / rsold * p
        rsold = rsnew
    else:
        if verbose:
            print('CG did not converge')
    return x

=======

def CG(x, b, A, max_iter, tol=1e-6):
    r = b - A @ x
    p = r
    rsold = r.T @ r
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if rsnew < tol:
            print('CG: {} iterations'.format(i))
            return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    print('CG: {} iterations'.format(max_iter))
    return x

=======

def CG(A, b, x0, tol, maxiter):
    x = x0
    r = b - A.matvec(x)
    p = r
    rTr = cp.dot(r, r)
    for i in range(maxiter):
        Ap = A.matvec(p)
        alpha = rTr/cp.dot(p, Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        rTr_new = cp.dot(r, r)
        if cp.sqrt(rTr_new) < tol:
            break
        beta = rTr_new/rTr
        p = r + beta*p
        rTr = rTr_new
    return x

=======

def CG(A, b, x, tol, maxIter, verbose=False):
    # CG Conjugate gradients
    #
    # CG(A, b, x, tol, maxIter) attempts to solve the system of linear
    # equations A*x = b for x.
    #
    # CG(A, b, x, tol, maxIter, verbose) also displays the residual
    # norm at each iteration.
    #
    # Input:
    # A - a symmetric positive definite matrix
    # b - the right-hand side vector
    # x - the initial guess
    # tol - the tolerance for the residual norm
    # maxIter - the maximum number of iterations
    # verbose - if true, displays the residual norm at each iteration
    #
    # Output:
    # x - the approximate solution
    # i - the number of iterations performed
    # rNorms - the residual norm at each iteration
    #
    # Adapted from https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    # Initialize
    r = b - A @ x
    p = r
    rNorms = []
    rNorms.append(cp.linalg.norm(r))

    # Main loop
    for i in range(maxIter):
        if verbose:
            print('Iteration %d, residual norm %.3e' % (i, rNorms[-1]))
        if rNorms[-1] < tol:
            break
        Ap = A @ p
        alpha = (r.T @ r) / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        beta = (r.T @ r) / (rNorms[-1] ** 2)
        p = r + beta * p
        rNorms.append(cp.linalg.norm(r))

    return x, i, rNorms

=======

def CG(cuA, cuc, cuX, maxIter=1000, tol=1e-6, verbose=False):
    """Conjugate gradients method for solving Ax = c
    
    Args:
        cuA (cupy.ndarray): A
        cuc (cupy.ndarray): c
        cuX (cupy.ndarray): initial guess
        maxIter (int, optional): maximum number of iterations. Defaults to 1000.
        tol (float, optional): tolerance. Defaults to 1e-6.
        verbose (bool, optional): print iteration number and residual norm. Defaults to False.
    
    Returns:
        cupy.ndarray: solution
    """
    cuR = cuc - cuA @ cuX
    cuP = cuR.copy()
    for i in range(maxIter):
        cuRTr = cuR.T @ cuR
        cuAlpha = cuRTr / (cuP.T @ cuA @ cuP)
        cuX += cuAlpha * cuP
        cuR -= cuAlpha * cuA @ cuP
        cuBeta = (cuR.T @ cuR) / cuRTr
        cuP = cuR + cuBeta * cuP
        if verbose:
            print(i, cupy.linalg.norm(cuR))
        if cupy.linalg.norm(cuR) < tol:
            return cuX
    return cuX


# Prompt: Conjugate Gradients pyCUDA\ndef
Synthesizing 10/10 solutions

=======

def CG(A,b,x0,tol,maxiter):
    import numpy as np
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.elementwise import ElementwiseKernel
    from pycuda.reduction import ReductionKernel
    from pycuda.compiler import SourceModule

    # Initialize
    r = b - A*x0
    p = r
    rsold = r.dot(r)
    rsnew = rsold

    # Iteration
    for i in range(maxiter):
        Ap = A*p
        alpha = rsold/np.dot(p,Ap)
        x0 = x0 + alpha*p
        r = r - alpha*Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x0

=======

def CG(A, b, x, tol, max_iter):
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            print("Converged after", i, "iterations")
            return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    print("Did not converge")
    return x

=======

def CG(A, b, x, tol, maxiter):
    r = b - A(x)
    p = r
    rsold = r.dot(r)

    for i in range(maxiter):
        Ap = A(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            return x, i
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, maxiter

=======

def CG(A, b, x0, tol=1e-5, maxiter=100):
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import numpy as np
    import time

    mod = SourceModule("""
    __global__ void CG(double *A, double *b, double *x, double *r, double *p, double *alpha, double *beta, double *r_new, double *z, double *Ap, double *tol, int *maxiter, int *iter, int *n)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int idx = i + j * n[0];

        if (i < n[0] && j < n[0])
        {
            r[idx] = b[idx] - A[idx] * x[idx];
            p[idx] = r[idx];
        }

        __syncthreads();

        if (i == 0 && j == 0)
        {
            r_new[0] = dot(r, r, n[0]);
        }

        __syncthreads();

        for (iter[0] = 0; iter[0] < maxiter[0]; iter[0]++)
        {
            if (i < n[0] && j < n[0])
            {
                Ap[idx] = A[idx] * p[idx];
            }

            __syncthreads();

            if (i == 0 && j == 0)
            {
                alpha[0] = r_new[0] / dot(p, Ap, n[0]);
            }

            __syncthreads();

            if (i < n[0] && j < n[0])
            {
                x[idx] += alpha[0] * p[idx];
                r[idx] -= alpha[0] * Ap[idx];
            }

            __syncthreads();

            if (i == 0 && j == 0)
            {
                z[0] = dot(r, r, n[0]);
            }

            __syncthreads();

            if (i == 0 && j == 0)
            {
                if (sqrt(z[0]) < tol[0])
                {
                    break

=======

def CG(A, b, x0, tol, maxiter):
    from pycuda import gpuarray
    from pycuda import driver
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import numpy as np

    # Initialize
    x = x0
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)

    # Start iteration
    for i in range(maxiter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

=======

def CG(x, A, b, tol=1e-6, max_iter=100):
    r = b - A * x
    p = r
    rsold = r * r
    for i in range(max_iter):
        Ap = A * p
        alpha = rsold / (p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r * r
        if rsnew < tol:
            return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol, maxIter):
    # Initialize
    x = x0
    r = b - A*x
    p = r
    rsold = r.dot(r)
    
    # Iterate
    for i in range(maxIter):
        Ap = A*p
        alpha = rsold/(p.dot(Ap))
        x = x + alpha*p
        r = r - alpha*Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x

=======

def CG(A, b, x, NumIter, tol, verbose):
    r = b - A(x)
    p = r
    rsold = r.dot(r)
    for i in range(NumIter):
        Ap = A(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x, tol, maxiter):
    """Solves the linear system Ax = b using the Conjugate Gradient method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Sparse matrix of the system.
    b : numpy.ndarray
        Right hand side of the system.
    x : numpy.ndarray
        Initial guess.
    tol : float
        Tolerance for the stopping criterion.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    x : numpy.ndarray
        Solution to the system.
    """

    r = b - A*x
    p = r
    rsold = r.T.dot(r)
    for i in range(maxiter):
        Ap = A*p
        alpha = rsold/(p.T.dot(Ap))
        x = x + alpha*p
        r = r - alpha*Ap
        rsnew = r.T.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew

    return x

=======

def CG(A, b, x, niter, tol, verbose=True):
    # Conjugate Gradients pyCUDA
    # A is the matrix
    # b is the RHS
    # x is the initial guess
    # niter is the number of iterations
    # tol is the tolerance
    # verbose is a flag to print the residual
    # x is the solution

    # Ax = b
    # x = A^{-1}b

    # r = b - Ax
    # p = r
    # x = x + alpha*p
    # r = r - alpha*A*p
    # beta = r'*r/(r0'*r0)
    # p = r + beta*p

    # Initialize the residual
    r = b - A*x
    p = r
    r0 = r
    r0 = r0.get()

    # Loop over the number of iterations
    for i in range(niter):
        # Compute alpha
        Ap = A*p
        alpha = (r.T@r)/(p.T@Ap)

        # Update the solution
        x = x + alpha*p

        # Update the residual
        r = r - alpha*Ap

        # Compute beta
        beta = (r.T@r)/(r0.T@r0)

        # Update the search direction
        p = r + beta*p

        # Print the residual
        if verbose:
            print('Residual:', np.linalg.norm(r))

        # Check the residual
        if np.linalg.norm(r) < tol:
            break

    # Return the solution
    return x

# Prompt: Conjugate Gradients numpy\ndef
Synthesizing 10/10 solutions

=======

def CG(A, b, x, tol, max_iter):
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            return x, i
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, i

=======

def CG(A, b, x0, tol, max_iter):
    x = x0
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol=1e-5, maxiter=100):
    x = x0
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    for i in range(maxiter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol=1e-6, maxiter=100):
    x = x0
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    for i in range(maxiter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            return x, i+1
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, maxiter

=======

def CG(A,b,x0,tol=1e-8):
    x = x0
    r = b - np.dot(A,x)
    p = r
    rsold = np.dot(r.T,r)
    for i in range(len(b)):
        Ap = np.dot(A,p)
        alpha = rsold/(np.dot(p.T,Ap))
        x = x + alpha*p
        r = r - alpha*Ap
        rsnew = np.dot(r.T,r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x

=======

def CG(A, b, x, maxiter=1000, tol=1e-6):
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    for i in range(maxiter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x0, tol=1e-6, maxiter=10000):
    """
    Conjugate Gradients method for solving Ax=b
    """
    x = x0
    r = b - np.dot(A, x)
    p = r.copy()
    rsold = np.dot(r.T, r)
    for i in range(maxiter):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

=======

def CG(A, b, x, tol=1e-8, max_iter=1000, verbose=True):
    r = b - A(x)
    p = r
    r_k_norm = np.dot(r,r)
    for i in range(max_iter):
        Ap = A(p)
        alpha = r_k_norm / np.dot(p,Ap)
        x += alpha * p
        r -= alpha * Ap
        r_kplus1_norm = np.dot(r,r)
        if np.sqrt(r_kplus1_norm) < tol:
            break
        beta = r_kplus1_norm / r_k_norm
        p = r + beta * p
        r_k_norm = r_kplus1_norm
    if verbose:
        print("CG: Iterations = %d, Residual = %g" % (i+1, np.sqrt(r_kplus1_norm)))
    return x

=======

def CG(A,b,x0,eps):
    x = x0
    r = A*x - b
    p = -r
    rTr = r.T*r
    while (rTr > eps):
        Ap = A*p
        alpha = rTr/(p.T*Ap)
        x = x + alpha*p
        r = r + alpha*Ap
        rTr_1 = rTr
        rTr = r.T*r
        beta = rTr/rTr_1
        p = -r + beta*p
    return x

=======

def CG(A, b, x0, tol=1e-8, max_iter=100):
    # Conjugate Gradients
    # A: positive definite matrix
    # b: vector
    # x0: initial guess
    # tol: tolerance
    # max_iter: maximum iterations
    x = x0
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < tol:
            return x
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x
