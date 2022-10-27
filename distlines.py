# Author: Petar Sarajcev, PhD (petar.sarajcev@fesb.hr)
# University of Split, FESB, Department of Power Engineering, R. Boskovica 32,
# HR21000, Split, Croatia.
"""
References:
[1] J. A. Martinez and F. Gonzalez-Molina, "Statistical evaluation
of lightning overvoltages on overhead distribution lines using neural
networks," in IEEE Transactions on Power Delivery, vol. 20, no. 3,
pp. 2219-2226, July 2005, doi: 10.1109/TPWRD.2005.848734.
[2] A. R. Hileman, "Insulation Coordination for Power Systems", CRC Press,
Boca Raton, FL, 1999.
"""
from cmath import nan
import numpy as np
import pandas as pd
from scipy import stats
from scipy import integrate


def egm(I, model='Love'):
    """
    Electrogeometric model of lightning attachment to transmission lines.

    Arguments
    ---------
    model: string
        Electrogeometric (EGM) model name from one of the following options: 
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', where AW
        stands for Armstrong & Whitehead, while BW means Brown & Whitehead.
    I: float
        Lightning current amplitude in kA.

    Returns
    -------
    rg: float
        Striking distance to ground in meters.
    Ag: float
        Parameter A from the select EGM model r = A * I^b in relation to ground.
    rc: float
        Striking distance to phase conductor in meters.
    Ac: float
        Parameter A from the select EGM model r = A * I^b in relation to phase
        conductor.

    Raises
    ------
    NotImplementedError
    """
    if model == 'Wagner':
        Ag = 14.2
        bg = 0.42
        Ac = 14.2
        bc = 0.42
    elif model == 'Young':
        Ag = 27.
        bg = 0.32
        Ac = 27.
        bc = 0.32
    elif model == 'AW':
        # Armstrong & Whitehead
        Ag = 6.
        bg = 0.8
        Ac = 6.7
        bc = 0.8
    elif model == 'BW':
        # Brown & Whitehead
        Ag = 6.4
        bg = 0.75
        Ac = 7.1
        bc = 0.75
    elif model == 'Love':
        Ag = 10.
        bg = 0.65
        Ac = 10.
        bc = 0.65
    elif model == 'Anderson':
        Ag = 8.
        bg = 0.65
        Ac = 8.
        bc = 0.65
    else:
        raise NotImplementedError('Model {} is not recognized.'.format(model))
    
    rg = Ag*I**bg
    rc = Ac*I**bc

    return rg, rc, Ag, bg


def max_shielding_current(I, h, y, sg, model='Love'):
    """
    Compute maximum shielding current of the transmission line.

    Arguments
    ---------
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    model: string
        Electrogeometric (EGM) model name from one of the following options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', where AW
        stands for Armstrong & Whitehead, while BW means Brown & Whitehead.

    Returns
    -------
    Igm: float
        Maximum shielding current of the line (kA).
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')

    rg, rc, A, b = egm(1., model)
    a = sg / 2.
    alpha = np.arctan(a/(h-y))
    gamma = rc/rg
    ko = 1. - gamma**2*np.sin(alpha)**2
    rgm = ((h+y)/(2.*ko))*(1. + np.sqrt(1. - ko*(1. + (a/(h+y))**2)))
    Igm = (rgm/A)**(1./b)

    return Igm


def exposure_distances(I, h, y, sg, model='Love'):
    """
    Compute exposure distances to shield wire(s) and phase conductors.

    Arguments
    ---------
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    model: string
        Electrogeometric (EGM) model name from one of the following options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', where AW
        stands for Armstrong & Whitehead, while BW means Brown & Whitehead.

    Returns
    -------
    Dg: float
        Exposure distance to shield wire(s) or ground surface (m).
    Dc: float
        Exposure distance to phase conductor (m).

    Raises
    ------
    ValueError
        Height of the phase conductor should not exceed that of the shield wire(s).
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')

    rg, rc, A, b = egm(I, model)
    igm = max_shielding_current(I, h, y, sg, model)

    if I > igm:
        Dc = 0.
        thetac = np.arcsin((rg-h)/rc)
        Dg = rc*np.cos(thetac)

    else:
        if rg <= y:
            theta = 0.
        else:
            theta = np.arcsin((rg-y)/rc)
        a = sg / 2.
        alpha = np.arctan(a/(h-y))
        beta = np.arcsin(np.sqrt(a**2 + (h-y)**2)/(2.*rc))
        Dc = rc*(np.cos(theta) - np.cos(alpha + beta))
        Dg = rc*np.cos(alpha - beta)

    return Dg, Dc


def striking_point(x0, I, h, y, sg, model='Love', shield=True):
    """
    Determine the striking point of lightning flash.

    Note
    ----
    Strike event is coded as follows:
        0 - direct stroke to shield wire
        1 - direct stroke to phase conductor (shielding failure)
        2 - indirect (near-by) stroke
        3 - direct stroke to phase conductor without shield wires
        4 - indirect (near-by) stroke without shield wires

    Arguments
    ---------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] in (m) from
        the distribution line.
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    model: string
        Electrogeometric (EGM) model name from one of the following options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', where AW
        stands for Armstrong & Whitehead, while BW means Brown & Whitehead.
    shield: bool
        Presence of shield wire (True/False).

    Returns
    -------
    strike: int
        Strike point code value (0 to 4).

    Raises
    ------
    NotImplementedError
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')
    if shield:
        # Shield wire is present at the transmission line
        Dg, Dc = exposure_distances(I, h, y, sg, model)
        if x0 <= sg/2. + Dg:
            strike = 0  # stroke to shield wire
        elif sg/2. + Dg < x0 <= sg/2. + Dg + Dc:
            strike = 1  # stroke to phase conductor
        elif x0 > sg/2. + Dg + Dc:
            strike = 2  # indirect (near-by) stroke
        else:
            # Impossible situation encountered
            raise NotImplementedError('Impossible situation encountered!')
    else:
        # There is NO shield wire
        rg, rc, A, b = egm(I, model)
        Dc = np.sqrt(rc**2 - (rg-y)**2)
        if x0 <= sg/2. + Dc:
            strike = 3  # stroke to phase conductor without shield wires
        elif x0 > sg/2. + Dc:
            strike = 4  # indirect (near-by) stroke without shield wires
        else:
            # Impossible situation encountered
            raise NotImplementedError('Impossible situation encountered!')

    return strike


def impedance(height, radius):
    """
    Wave impedance of the single conductor above earth.

    Parameters
    ----------
    height: float
        Conductor height above earth (m).
    radius: float
        Conductor radius (m).

    Returns
    -------
    Z: float
        Wave impedance (Ohm).
    """
    Z = 60. * np.log((2. * height) / radius)
    return Z


def phase_conductor(I, y, rad_c):
    """
    Direct stroke to phase conductor with or without shield wire(s).

    Parameters
    ----------
    I: float A
        Lightning current amplitude (kA).
    y: float
        Phase conductor height (m).
    rad_c: float
        Phase conductor radius (m).

    Returns
    -------
    Volt: float
        Overvoltage amplitude (kV).
    """
    Zc = impedance(y, rad_c)  # phase conductor's wave impedance
    Volt = Zc * (I/2.)
    return Volt


def indirect_stroke(x0, I, y, v, c):
    """
    Indirect stroke (transmission line without shield wire).

    Arguments
    ---------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] in (m) from
        the distribution line.
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    v: float
        Velocity of the lightning return stroke (m/us).
    c: float
        Speed of light in free space (m/us).

    Returns
    -------
    Vc: float
        Overvoltage amplitude (kV).
    """
    k = ((30.*I*y)/x0)
    Vc = k * (1. + (1./np.sqrt(2.)) * (v/c) * (1. / np.sqrt(1. - 0.5*(v/c)**2)))
    return Vc


def impedances(h, y, sg, rad_s):
    """
    Wave impedance of the shield wire, including the mirror image.

    Parameters
    ----------
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    rad_s: float
        Radius of the shield wire (m).

    Returns
    -------
    Zsw, Zswc: float, float
        Wave impedances of the shield wire, including the mirror image in (ohm).

    Raises
    ------
    ValueError
        Height of the phase conductor should not exceed that of the shield wire(s).
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')

    Zsw = impedance(h, rad_s)  # shield wire's wave impedance
    a = sg / 2.
    Zswc = 60. * np.log(np.sqrt(a**2 + (h + y)**2) / np.sqrt(a**2 + (h - y)**2))
    return Zsw, Zswc


def indirect_shield_wire_present(x0, I, h, y, sg, v, c, R, rad_s):
    """
    Indirect stroke with TL shield wire present on the tower.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] in (m) from
        the distribution line.
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    v: float
        Velocity of the lightning return stroke (m/us).
    c: float
        Speed of light in free space (m/us).
    R: float
        Grounding resistance of the shield wire (Ohm).
    rad_s: float
        Radius of the shield wire (m).

    Returns
    -------
    Volt: float
        Overvoltage amplitude (kV).

    Raises
    ------
    ValueError
        Height of the phase conductor should not exceed that of the shield wire(s).    
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')

    Vc = indirect_stroke(x0, I, y, v, c)
    Zsw, Zswc = impedances(h, y, sg, rad_s)
    pr = 1. - (h/y) * (Zswc / (Zsw + 2.*R))
    Volt = pr * Vc
    return Volt


def indirect_chowdhuri_gross(x0, y, I, tf, h_cloud=3000., W=300., x=0.):
    """
    Chowdhuri-Gross model of nearby indirect lightning strike to
    distribution line without the shield wire.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] in (m) from
        the distribution line.
    y: float
        Height of the phase conductor (m).
    I: float
        Lightning current amplitude (kA).
    tf: float
        Lightning current wavetime front duration (us).
    h_cloud: float
        Cloud base height (m).
    W: float
        Lightning return stroke speed (m/us).
    x: float
        Distance on the line where the overvoltage will be computed (m).
        Distance of x = 0 is the location on the line that is exactly 
        perpendicular to the lightning strike.

    Returns
    -------
    Vmax: float
        Peak value of the overvoltage at the selected location (x) on the line.
    V, ti: 1d-arrays
        Overvoltage values and associated time instances, respectively.
    """
    # Convert for computation
    I = I * 1e3     # kA => A
    tf = tf * 1e-6  # us => s
    # Additional data (fixed values)
    c = 3e8   # speed of light in free space
    Time = 100e-6  # (s)
    dt = 0.1e-6    # (s) computational time step

    velocity = c / np.sqrt(1. + (W/(I * 1e-3)))
    beta = velocity / c

    N = int(Time/dt)
    V = np.empty(N)
    ti = np.empty(N)
    t = 0.
    for i in range(N):
        t0 = np.sqrt(x**2 + x0**2)/c
        b0 = 1. - beta**2
        t0f = t0 + tf
        ttf = t - tf

        s1 = ((c*t0 - x)**2 + x0**2)**2
        s2 = (4.*h_cloud**2) * (c*t0-x)**2
        m0 = np.sqrt(s1 + s2)

        s1 = ((c*t - x)**2 + x0**2)**2
        s2 = (4.*h_cloud**2) * (c*t - x)**2
        m1 = np.sqrt(s1 + s2)
        
        s1 = ((c*t0 + x)**2 + x0**2)**2
        s2 = (4.*h_cloud**2) * (c*t0 + x)**2
        n0 = np.sqrt(s1 + s2)

        s1 = ((c*t + x)**2 + x0**2)**2
        s2 = (4.*h_cloud**2) * (c*t + x)**2
        n1 = np.sqrt(s1 + s2)

        s1 = ((c*ttf - x)**2 + x0**2)**2
        s2 = (4.*h_cloud**2) * (c*ttf - x)**2
        m1a = np.sqrt(s1 + s2)

        s1 = ((c*ttf + x)**2 + x0**2)**2
        s2 = (4.*h_cloud**2) * (c*ttf + x)**2
        n1a = np.sqrt(s1 + s2)

        f0 = (30.*I*y) / (tf*beta*c)
        f1 = m1 + (c*t - x)**2 - x0**2
        f2 = m1 - (c*t - x)**2 + x0**2
        f3 = m0 + x0**2 - (c*t0 - x)**2
        f4 = m0 - x0**2 + (c*t0 - x)**2
        f5 = n1 + (c*t + x)**2 - x0**2
        f6 = n1 - (c*t + x)**2 + x0**2
        f7 = n0 + x0**2 - (c*t0 + x)**2
        f8 = n0 - x0**2 + (c*t0 + x)**2

        f9 = b0 * (beta**2*x**2 + x0**2) + beta**2*c**2*t**2 * (1. + beta**2)
        f10 = (2.*beta**2*c*t) * np.sqrt(beta**2*c**2*t**2 + b0*(x**2 + x0**2))
        f11 = (c**2*t**2 - x**2) / x0**2
        f12 = (f9 - f10) / (b0**2*x0**2)
        f13 = (f1*f3*f5*f7) / (f2*f4*f6*f8)

        f1a = m1a + (c*ttf - x)**2 - x0**2
        f2a = m1a - (c*ttf - x)**2 + x0**2
        f3a = f3
        f4a = f4
        f5a = n1a + (c*ttf + x)**2 - x0**2
        f6a = n1a - (c*ttf + x)**2 + x0**2
        f7a = f7
        f8a = f8

        f9a = b0*(beta**2*x**2 + x0**2) + (beta**2*c**2*ttf**2) * (1. + beta**2)
        f10a = (2.*beta**2*c*ttf) * np.sqrt(beta**2*c**2*ttf**2 + b0*(x**2 + x0**2))
        f11a = (c**2*ttf**2 - x**2) / x0**2
        f12a = (f9a - f10a) / (b0**2*x0**2)
        f13a = (f1a*f3a*f5a*f7a) / (f2a*f4a*f6a*f8a)
        
        if (t < t0):
            V1 = 0.
        else:
            FF1 = f0 * (b0*np.log(f12) - b0*np.log(f11) + 0.5*np.log(f13))
            V1 = FF1
        if (t < t0f):
            V2 = 0.
        else:
            FF2 = -f0 * (b0*np.log(f12a) - b0*np.log(f11a) + 0.5*np.log(f13a))
            V2 = FF2

        V[i] = (V1 + V2) * 1e-3  # kV
        ti[i] = t*1e6  # us
        
        t = t + dt

    Volt = abs(V)
    Vmax = max(Volt)
    
    return Vmax, ti, V


def backflashover(I, h, y, sg, c, Ri, rad_s, span_length=150.):
    """
    Lightning strike to shield wire and the backflashover overvoltage.

    Parameters
    ----------
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    c: float
        Speed of light in free space (m/us).
    Ri: float
        Stricken tower's impulse grounding resistance (Ohm).
    rad_s: float
        Radius of the shield wire (m).
    span_length: float
        Span length (default value: 150 m on distribution lines).

    Returns
    -------
    Volt: float
        Overvoltage amplitude (kV).

    Raises
    ------
    ValueError
        Height of the phase conductor should not exceed that of the shield
        wire(s).
    
    Notes
    -----
    Analysis of the backflashover phenomenon is presented in detail in the
    following reference:
        A. R. Hileman, Insulation Coordination for Power Systems, CRC Press,
        Boca Raton (FL), 1999, pp. 373-423.
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')

    Zsw, Zswc = impedances(h, y, sg, rad_s)
    Z = Zsw/2.
    Rn = Ri/2.  # grounding resistance of the tower at the other side of the
    # span it is assumed here that it equals half of the (impulse) grounding
    # impedance of the stricken transmission line tower; this assumption does
    # not have significant impact on the outcome.
    Zw = ((2.*Ri**2*Z) / (Z + Ri)**2) * ((Z - Rn) / (Z + Rn))
    Zi = (Ri*Z) / (Z + Ri)
    Fi = ((Z - Ri) / (Z + Ri)) * ((Z - Rn) / (Z + Rn))
    Tau = span_length / c
    T = 2.  # fixed; see the original paper for more info.
    N = int(np.floor(T / (2.*Tau)))
    kv = I * Tau * Zw * ((1. - Fi**N) / (1. - Fi)**2 - (float(N)*Fi**N) / (1. - Fi))
    Vt = 0.5 * I * T * (Zi - (Zw * (1. - Fi**N)) / (1. - Fi)) + kv
    CF = Zswc / Zsw
    Volt = Vt * (1. - CF)

    return Volt


def backflashover_cigre(I, Un, R0, rho, h, rad_s, 
                        span_length=150., CFO=150., KPF=0.7,
                        C=0.35, Eo=400., eps_Ri=0.1):
    """
    Simplified CIGRE method for calculating backflashover on overhead
    transmission (distribution) lines.

    Parameters
    ----------
    I: float
        Lightning current amplitude (kA).
    Un: float
        Nominal voltage of the line (kV).
    R0: float
        Grounding resistance (at low-current level) of the distribution
        line's tower (Ohm). Typical values range between 10 to 50 Ohms.
    rho: float
        Average soil resistivity at the tower's site (Ohm/m). Parameters
        `rho` and `R0` define the factor `rho/R0` which is typically found
        in the range between 10 and 50.
    CFO: float
        Critical flashover voltage level of the line's insulation (kV).
    KPF: float
        Dimensionless factor (so-called power frequency factor) for
        correcting the nominal voltage of the line. For a horizontal
        configuration of the line the recommended value is 0.7, while
        for a vertical configuration the recommended value is 0.4.
    span_length: float
        Average length of a single span of the distribution line (m).
    h: float
        Height of the shield wire (m).
    rad_s: float
        Shield wire radius (m).
    C: float
        Coupling factor (dimensionless) between the shield wire(s) and the
        phase conductors. Recommended average value of this factor is 0.35.
    Eo: float
        Electric field strength for the inception of the soil ionization
        at the tower grounding (kV/m). Recommended value is 400 kV/m.
    eps_Ri: float
        Tolerance of the tower's grounding impulse impedance value for
        terminating the iterative computation procedure of the simplified
        method.
    
    Returns
    -------
    Ic: float
        Critical current value for the onset of the backflashover event, (kA).
    Vc: float
        Backflashover overvoltage value across the insulator strings on the
        stricken tower, (kV).
    
    Raises
    ------
    Exception
        General exception is raised if the while loop exhausts the counter
        without convergence.
    
    Notes
    -----
    This method is described in detail (including its underlying assumptions
    and limitations) in the following reference:
        A. R. Hileman, Insulation Coordination for Power Systems, CRC Press,
        Boca Raton (FL), 1999, pp. 373-423.
    """
    # Power frequency phase voltage
    VPF = KPF * (Un*np.sqrt(2.)/np.sqrt(3.))
    # Travel time of the single span (us)
    Ts = span_length / 300.  # c = 300 m/us
    # Surge impedance of the shield wire(s)
    Zg = impedance(h, rad_s)
    
    k = 0
    Ri = R0/2.
    while k < 10000:
        Tau = (Zg/Ri)*Ts
        CFOns = (0.977 + 2.82/Tau)*(1. - 0.2*VPF/CFO)*CFO
        Re = (Ri*Zg)/(Zg + 2.*Ri)
        Ic = (CFOns - VPF)/((1. - C)*Re)
        IR = (Re/Ri)*Ic
        Ig = (rho*Eo)/(2.*np.pi*R0**2)
        Rin = R0/np.sqrt(1. + IR/Ig)
        # Test for convergence
        if abs(Rin - Ri) <= eps_Ri:
            break
        else:
            Ri = Rin
        k += 1
    else:
        raise Exception('Error: Iterative method did not converge!')    
    
    # Backflashover overvoltage
    Vc = (1. - C) * Re * I
    return Ic, Vc


def induced_overvoltage(x0, I, h, y, sg, w, Ri, rad_c, rad_s, R, c,
                        model='Love', shield=True, span_length=150.):
    """
    Compute induced overvoltage on transmission line for different
    types of lightning strokes, where strike event has been coded
    as follows:
        0 - direct stroke to shield wire
        1 - direct stroke to phase conductor (shielding failure)
        2 - indirect (near-by) stroke where shield wires are present
        3 - direct stroke to phase conductor without shield wires
        4 - indirect (near-by) stroke where shield wires are absent

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] in (m) from
        the distribution line.
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    w: float
        Lightning return stroke velocity (m/us).
    Ri: float
        Tower's (impulse) resistance/ impedance (Ohm).
    rad_c: float
        Phase conductor radius (m).
    rad_s: float
        Shield wire radius (m).
    R: float
        Grounding resistance of shield wire (Ohm).
    c: float
        Speed of light in free space (300 m/us).
    model: string
        Electrogeometric (EGM) model name from one of the following options: 
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', where AW
        stands for Armstrong & Whitehead, while BW means Brown & Whitehead.
    shield: bool
        Presence of shield wire (True/False).
    span_length: float
        Span length (default value: 150 m on distribution lines).

    Returns
    -------
    return: float
        Induced voltage amplitude (kV).
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')

    # Compute return-stroke velocity
    v = c/np.sqrt(1. + w/I)

    # Point of lightning strike
    stroke = striking_point(x0, I, h, y, sg, model, shield)

    # Compute overvoltage on the transmission line
    if shield:
        # Shield wire is present on the transmission line
        if stroke == 1:
            # Stroke to phase conductor
            V = phase_conductor(I, y, rad_c)
        elif stroke == 2:
            # indirect stroke
            V = indirect_shield_wire_present(x0, I, h, y, sg, v, c, R, rad_s)
        elif stroke == 0:
            # stroke to shield wire (with backflashover)
            V = backflashover(I, h, y, sg, c, Ri, rad_s, span_length)
        else:
            # Impossible situation encountered
            raise NotImplementedError('Impossible situation encountered!')
    else:
        # There is NO shield wire on the transmission line
        if stroke == 3:
            # Stroke to phase conductor
            V = phase_conductor(I, y, rad_c)
        elif stroke == 4:
            # indirect stroke
            V = indirect_stroke(x0, I, y, v, c)
        else:
            # Impossible situation encountered
            raise NotImplementedError('Impossible situation encountered!')
    return V


def flashover(x0, I, h, y, sg, w, Ri, CFO, model='Love', shield=True, **params):
    """
    Determine if the flashover has occurred or not.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] in (m) from
        the distribution line.
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    w: float
        Lightning return stroke velocity (m/us).
    Ri: float
        Tower's (impulse) resistance/ impedance (Ohm).
    CFO: float
        Critical flashover voltage level (kV).
    rad_c: float
        Phase conductor radius (m).
    rad_s: float
        Shield wire radius (m).
    R: float
        Grounding resistance of shield wire (Ohm).
    c: float
        Speed of light in free space (300 m/us).
    model: string
        Electrogeometric (EGM) model name from one of the following options: 
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', where AW
        stands for Armstrong & Whitehead, while BW means Brown & Whitehead.
    shield: bool
        Presence of shield wire (True/False).

    return: float
        Flashover (0/1) indicator.
    """
    from operator import itemgetter

    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')
    # Unpacking extra arguments
    rad_s, rad_c, R, c, span_length = itemgetter(
        'rad_s', 'rad_c', 'R', 'c', 'span_length')(params)

    # Compute induced overvoltage
    overvoltage = induced_overvoltage(x0, I, h, y, sg, w, Ri, rad_c, rad_s,
                                      R, c, model, shield, span_length)
    # Determine if there is a flashover or not
    if abs(overvoltage) > CFO:
        flash = True
    else:
        flash = False
    return flash


def transmission_line(N, h, y, sg, distances, amplitudes, w, Ri,
                      egm_models, shield_wire, CFO=150.,
                      rad_c=5e-3, rad_s=2.5e-3, R=10., c=300.,
                      span_length=150.):
    """
    Determine if the flashover has occurred or not for any transmission line
    This subroutine calls "flashover" subroutine for each set of random values
    Note: Typical distribution line geometry is used for default values.

    N: int
        Number of flashover simulations.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    distances: array
        Perpendicular distances from the strike point to line (m).
    amplitudes: array
        Lightning current amplitudes (kA).
    w: array
        Lightning return stroke velocity (m/us).
    Ri: array
        Tower's (impulse) resistance/ impedance (Ohm).
    egm_model: list of strings
        Electrogeometric (EGM) model name from one of the following options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', where AW
        stands for Armstrong & Whitehead, while BW means Brown & Whitehead.
    shield_wire: list of bools
        Presence of shield wire (True/False).
    CFO: float
        Critical flashover voltage level (kV).
    rad_c: float
        Phase conductor radius (m).
    rad_s: float
        Shield wire radius (m).
    R: float
        Grounding resistance of shield wire (Ohm).
    c: float
        Speed of light in free space (300 m/us).
    span_length: float
        Span length (default value: 150 m on distribution lines).

    return: array of bools 
        Flashover (0/1) indicator.
    """
    if y > h:
        raise ValueError('y > h: Height of the phase cond. (y) should NOT exceed'
                         ' that of the shield wire (h).')
    # Default parameters
    params = {'rad_s': rad_s, 'rad_c': rad_c, 'R': R, 'c': c, 
              'span_length': span_length}

    # Flashover computation
    flash = np.empty_like(amplitudes)
    for j in range(N):
        flash[j] = flashover(distances[j], amplitudes[j], h, y, sg, w[j], Ri[j],
                             CFO, model=egm_models[j], shield=shield_wire[j],
                             **params)
    return flash


def generate_samples(N, XMAX=500, RiTL=50., Imu=31., sigmaI=0.55):
    """
    Generate random samples for the Monte Carlo simulation.

    Parameters
    ----------
    N: int
        Number of random samples to generate.
    XMAX: float
        Max. strike distance from the TL (m).
    RiTL: float
        Mean value of the tower's grounding impulse resistance (OHM).
    Imu: float
        Median value of lightning current amplitudes statistical
        distribution (kA).
    sigmaI: float
        Standard deviation of lightning current amplitudes statistical
        distribution. Default values for `Imu` and `sigmaI` are taken from
        the IEC 62305 and IEC 60071.

    Returns
    -------
    return: arrays
        Random samples from the appropriate statistical distributions of:
        amplitudes, return stroke velocities, stroke distances, tower
        grounding impedances, shield wire indicators, and EGM models.

    Note
    ----
    Routine returns following random samples: amplitudes, return stroke 
    velocities, distances of lightning strikes from line, impulse impedances
    of the line tower, shield wire(s) presence or absence indicators, and EGM
    models variants.
    """
    # Lightning current amplitudes (IEC 62305)
    I = stats.lognorm(s=sigmaI, loc=0., scale=Imu).rvs(size=N)
    
    # Return stroke velocity
    w = np.random.uniform(low=50., high=500., size=N)
    
    # Distance of lightning stroke from the transmission line
    distances = np.random.uniform(low=0., high=XMAX, size=N)
    
    # Tower grounding resistance
    Ri = stats.norm(loc=RiTL, scale=0.25*RiTL).rvs(size=N)
    Ri = np.where(Ri <= 0., RiTL, Ri)  # must be positive
    
    # Presence or absence of the shield wire(s)
    shield_wire = stats.bernoulli(p=0.5).rvs(size=N)  # wire present in 50% of cases
    
    # Select EGM models according to the custom probability levels
    egm_models_all = ['Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson']
    probabilities = [0.1, 0.2, 0.1, 0.1, 0.3, 0.2]  # custom levels
    egm_models = np.random.choice(egm_models_all, size=N, replace=True,
                                  p=probabilities)
    
    return I, w, distances, Ri, shield_wire, egm_models


def generate_dataset(N, h, y, sg, cfo, XMAX=500., RiTL=50., 
                     Imu=31., sigmaI=0.55, export=False):
    """
    Generating a random dataset of lightning flashovers on medium voltage
    distribution lines, by means of the Monte Carlo simulation of different
    possible flashover incidents (from both direct and indirect lightning
    strikes).

    Parameters
    ----------
    N: int
        Number of simulations for each distribution line geometry.
    h: array
        Array of shield wire heights, one entry for each distribution line, in
        meters.
    y: array
        Array of phase conductor heights, one entry for each distribution
        line, in meters.
    sg: array
        Array of distance values between shield wires, one entry for each
        distribution line, in meters.
    cfo: array
        Array of critical flashover voltage values, one entry for each
        distribution line, in kV.
    XMAX: float
        Maximum perpendicular distance from the distribution line for which
        lightning flashover interaction is considered feasible, in meters.
    RiTL: float
        Average value of the tower's grounding impedance (used as a mean
        value in the appropriate Normal distribution), in Ohm.
    Imu: float
        Median value of lightning current amplitudes statistical distribution,
        in kA.
    sigmaI: float
        Standard deviation of lightning current amplitudes statistical
        distribution. Default values for `Imu` and `sigmaI` are taken from the
        IEC 62305 and IEC 60071.
    export: bool
        Indicator True/False for exporting generated dataset into the CSV
        format.

    Returns
    -------
    data: pd.DataFrame
        Randomly generated dataset of lightning flashovers on distribution lines.
    
    Notes
    -----
    Dataset considers several distribution lines at the same time, where each
    line is of the same type, but with a different geometry. Each line has a
    flat configuration of phase conductors (at the height `y`), and with a 
    double shield wires (at the height `h`) that are seprated by distance `sg`.
    Each line can also have a different critical flashover voltage (CFO) value.
    """
    data = {'dist': [], 'ampl': [], 'shield': [], 'veloc': [], 
        'Ri': [], 'EGM': [], 'CFO': [], 'height': [], 'flash': []}

    for j in range(y.size):
        height = np.repeat(y[j], N)
        cfo_value = np.repeat(cfo[j], N)
        # Generate random samples
        amps, w, dists, Ri, sws, egms = generate_samples(
            N, XMAX, RiTL, Imu, sigmaI)
        
        # Simulate flashovers
        f = transmission_line(N, h[j], y[j], sg[j], 
                              dists, amps, w, Ri, egms, sws, CFO=cfo[j])
        
        # Store data as dict
        data['dist'].append(dists)
        data['ampl'].append(amps)
        data['shield'].append(sws)
        data['veloc'].append(w)
        data['Ri'].append(Ri)
        data['EGM'].append(egms)
        data['CFO'].append(cfo_value)
        data['height'].append(height)
        data['flash'].append(f)

    # Transfer data from dictionary to pandas DataFrame
    dataset = {}
    for key, value in data.items():
        dataset[key] = np.array(value).flatten()
    data = pd.DataFrame(data=dataset)

    if export:
        # Export data to csv
        data.to_csv('distlines.csv')

    return data


def lightning_amplitudes_pdf(x, mu=31., sigma=0.55):
    """ 
    Probability density function (PDF) of the lightning 
    current amplitudes Log-Normal statistical distribution.

    Parameters
    ----------
    x: float
        Value of the lightning current amplitude (kA) at which the
        Log-Normal distribution is to be evaluated.
    mu: float
        Median value of the Log-Normal distribution of lightning
        current amplitudes, (kA).
    sigma: float
        Standard deviation of the Log-Normal distribution of
        lightning current amplitudes.
    
    Returns
    -------
    pdf: float
        Probability density function (PDF) value.

    Note
    ----
    Default values for the median value and standard deviation of the
    Log-Normal distribution (of lightning current amplitudes) has been
    taken from the relevant CIGRE/IEEE WG recommendations (see also
    IEC 60071 for additional information).
    """
    denominator = (np.sqrt(2.*np.pi)*x*sigma)
    pdf = np.exp(-(np.log(x) - np.log(mu))**2 / (2.*sigma**2)) / denominator
    # Convert `nan` to numerical values
    pdf = np.nan_to_num(pdf)

    return pdf


def risk_of_flashover(support, y_hat, method='simpson'):
    """
    Compute risk of flashover with a numerical integration routine.

    Parameters
    ----------
    support: array
        Sample points at which the function to be integrated is defined.
        It is advisable to have odd number of sample points.
    y_hat: array
        Function values that are integrated at the support.
    method: string
        Integration method to use: 'simpson' or 'trapezoid'.

    Returns
    -------
    risk: float
        Risk of insulation flashover.
    
    Raises
    ------
    NotImplementedError
    
    Note
    ----
    For an odd number of samples that are equally spaced the result od simpson's
    method is exact if the function is a polynomial of order 3 or less.
    """
    pdf = lightning_amplitudes_pdf(support)
    integrand = pdf * y_hat

    if method == 'simpson':
        # Simpson's rule
        risk = integrate.simpson(integrand, support)
    elif method == 'trapezoid':
        # Trapezoid rule
        risk = integrate.trapezoid(integrand, support)
    else:
        raise NotImplementedError('Method {} not recognized!'.format(method))
    
    return risk


def risk_curve_fit(x, a, b):
    """ 
    Function for the least-squares fit of the relationship 
    between statistical safety factor and a risk of flashover.

    Parameters
    ----------
    x: array
        Sample points on the x-axis (safety factor values).
    a, b: float
        Coefficients of the function.
    
    Returns
    -------
    y: array
        Points on the y-axis (risk values).
    """
    return a * np.exp(-b * x)


def jitter(ax, x, y, s, c, **kwargs):
    """ Add jitter to the scatter plot.

    Parameters
    ----------
    ax: ax
        Matplotlib axis object.
    x: array
        Array of x-values.
    y: array
        Array of y-values.
    s: int
        Marker/dot size.
    c: string
        Marker/dot color.

    Returns
    -------
    return:
        ax.scatter matplotlib object.
    """
    def random_jitter(arr, std=0.01):
        from numpy.random import randn

        stdev = std * (max(arr) - min(arr))
        return arr + randn(len(arr)) * stdev

    return ax.scatter(random_jitter(x), random_jitter(y), s=s, c=c, **kwargs)


#   *** MAIN PROGRAM ***
if __name__ == "__main__":
    """ This is the main program."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Figure aesthetic
    sns.set(context='paper', style='whitegrid')
    sns.set_style('ticks', {'xtick.direction': 'in', 'ytick.direction': 'in'})

    # Number of random samples
    N = 1000
    # Generate random samples for the Monte Carlo simulation
    # the same samples are used for all transmission lines
    amps, w, dists, Ri, sws, egms = generate_samples(N)

    # Transmission line geometry (single line example)
    h = 11.5  # shield wire height (m)
    y = 10.   # phase conductor height (m)
    sg = 3.   # distance between shield wires (m)

    # Flashover analysis for a single transmission line
    fl = transmission_line(N, h, y, sg, dists, amps, w, Ri, egms, sws)

    # Graphical visualization of simulation results
    # marginal of distance
    fig, ax = plt.subplots(figsize=(7, 5))
    jitter(ax, dists[sws == True], fl[sws == True], s=20,
           c='darkorange', label='shield wire')
    jitter(ax, dists[sws == False], fl[sws == False], s=5,
           c='royalblue', label='NO shield wire')
    ax.legend(loc='center right')
    ax.set_ylabel('Flashover probability')
    ax.set_xlabel('Distance (m)')
    ax.grid()
    plt.show()
    # marginal of amplitude
    fig, ax = plt.subplots(figsize=(7, 5))
    jitter(ax, amps[sws == True], fl[sws == True], s=20,
           c='darkorange', label='shield wire')
    jitter(ax, amps[sws == False], fl[sws == False], s=5,
           c='royalblue', label='NO shield wire')
    ax.legend(loc='center right')
    ax.set_ylabel('Flashover probability')
    ax.set_xlabel('Amplitude (kA)')
    ax.grid()
    plt.show()
    # in two dimensions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dists[fl == 0], amps[fl == 0], s=20,
               color='darkorange', label='NO flashover')
    ax.scatter(dists[fl == 1], amps[fl == 1], s=20,
               color='royalblue', label='flashover')
    ax.legend(loc='upper right')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Amplitude (kA)')
    ax.grid()
    plt.show()
