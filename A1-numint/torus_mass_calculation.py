import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


def f(t):
    """
    Integrand function for the mass of a torus in terms of t, 
    derived from washer method.
    """
    return np.sqrt(1 - t**2)


def newton_raphson(f, df, x0, tol=1e-6):
    """
    Newton-Raphson method to find the roots of a function f(x).
    """
    x = x0
    while np.abs(f(x)) > tol:
        x = x - f(x)/df(x)
    return x

def secant(f, x0, x1, tol=1e-6):
    """
    Secant method to find the roots of a function f(x).
    """
    while np.abs(f(x1)) > tol:
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    return x1

def legendre_polynomial(n):
    """Compute the Legendre polynomial P_n(x)"""
    
    if n == 0:
        return np.poly1d([1])
    elif n == 1:
        return np.poly1d([1, 0])

    Pn_2 = np.poly1d([1])     # initialize P_{n-2}(x) = 1
    Pn_1 = np.poly1d([1, 0])  # and P_{n-1}(x) = x

    # Recurrence relation
    # The iterative approach is more efficient than the recursive one
    for k in range(2, n+1):
        Pn = ((2*k - 1) * np.poly1d([1, 0]) * Pn_1 - (k - 1) * Pn_2) / k
        Pn_2, Pn_1 = Pn_1, Pn  # Update recurrence relation terms
    
    return Pn


def gauss_legendre(n, func=f):
    """
    Compute the integral of a function over the interval [-1, 1] 
    using Gauss-Legendre quadrature of order n.
    
    Parameters: 
    n: int, order of the quadrature
    func: callable, function to be integrated

    Returns:
    float, the integral of the function
    array, nodes
    array, weights
    """

    # Get Legendre polynomial P_n(x) and its derivative P'_n(x)
    Pn = legendre_polynomial(n)
    Pn_deriv = np.polyder(Pn)

    # Compute the roots of P_n(x) with Newton-Raphson method.
    # Using the nodes of the Chebyshev polynomial as initial guesses
    initial_guesses = np.cos((2*np.arange(1, n+1) - 1) * np.pi / (2*n))
    nodes = np.array([newton_raphson(Pn, Pn_deriv, x) for x in initial_guesses])

    # Compute the weights
    w = 2 / ((1-nodes**2) * (Pn_deriv(nodes)**2))
    
    return np.sum(w*func(nodes)), nodes, w


def bode_rule(n, func=f):
    """ 
    Compute the integral of a function over the interval [-1, 1]
    using the composite Bode's rule of order n.

    Parameters:
    n: int, number of subintervals. Must be multiple of 4.
    f: callable, function to be integrated

    Returns:
    float, the integral of the function
    array, nodes
    array, weights
    """
    x = np.linspace(-1, 1, n + 1)
    y = func(x)
    h = (x[-1] - x[0]) / n
    
    weights = np.zeros_like(x)
    weights[::n] = 7
    weights[1:-1:2] = 32 
    weights[2:-2:4] = 12 
    weights[4:-4:4] = 14
    weights *= 2 * h / 45

    integral = np.sum(weights * y)

    return integral, x, weights


def main():
    # Parameters
    rho = 1000
    r_outer = 0.15
    r_inner = 0.1
    R = (r_outer + r_inner)/2
    r = (r_outer - r_inner)/2
    C = 4*np.pi*rho*R*r**2

    # Exact mass of the torus
    M_exact = rho * 2 * np.pi**2 * R * r**2

    glq_error = []
    simpson_error = []
    bode_error = []
    n_values = range(2, 37, 2) 

    # The torus mass is given by C * ∫f(t)dt, over the interval [-1, 1].
    # Compute the error of the Gauss-Legendre quadrature, Simpson's rule
    # and Bode's rule for different values of n.
    for n in n_values:
        x = np.linspace(-1, 1, n+1)

        glq_error.append(np.abs(C*gauss_legendre(n)[0] - M_exact))
        simpson_error.append(np.abs(C*spi.simpson(f(x), x) - M_exact))
        if n%4 == 0: # n must be multiple of 4 for Bode's rule
            bode_error.append(np.abs(C*bode_rule(n)[0] - M_exact))

    print(glq_error[-1])
    plt.figure()
    plt.plot(n_values, glq_error, marker='o', label='Gauss-Legendre')
    plt.plot(n_values, simpson_error, marker='s', label='Simpson')
    plt.plot([n for n in n_values if n % 4 == 0], bode_error, marker='^', label="Bode")
    plt.yscale('log')
    plt.xlabel('Number of Points')
    plt.ylabel('Error (kg)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

