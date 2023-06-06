# Author: Petar Sarajcev, PhD (petar.sarajcev@fesb.hr)
# University of Split, FESB, Department of Power Engineering, R. Boskovica 32,
# HR21000, Split, Croatia.
"""
References:
[1] J. A. Martinez and F. Gonzalez-Molina, "Statistical evaluation
    of lightning overvoltages on overhead distribution lines using 
    neural networks," in IEEE Transactions on Power Delivery, vol. 
    20, no. 3, pp. 2219-2226, July 2005, doi: 10.1109/TPWRD.2005.848734.
[2] A. R. Hileman, "Insulation Coordination for Power Systems", 
    CRC Press, Boca Raton, FL, 1999.
[3] P. Chowdhuri, "Electromagnetic Transients in Power Systems", 
    Research Studies Press Ltd., Taunton, Somerset (UK), 1996.
[4] P. Chowdhuri, Analysis of lightning induced voltages on overhead
    lines, IEEE Transactions on Power Delivery, Vol. 4, No. 1, 1989, 
    pp. 479-492.
[5] A. C. Liew and S. C. Mar, Extension of the Chowdhuri-Gross model 
    for lightning induced voltage on overhead lines, IEEE Transactions 
    of Power Systems, Vol. PWRD-1, No. 2, 1986, pp. 240-247.
"""
import numpy as np
import pandas as pd


def egm_distance(I, A, b):
    """
    EGM striking distance.

    Electrogeometric (EGM) model's striking distance computation,
    of the form: r = A * I**b.

    Arguments
    ---------
    I: float
        Lightning current amplitude in kA.
    A, b: floats
        Parameters of the electrogeometric model.

    Returns
    -------
    r: float
        Distance in meters from the electrogeometric model.
    """
    if I < 0:
        I = abs(I)

    if A < 0 or b < 0:
        raise ValueError('Values A and b must be positive numbers.')

    r = A * I**b

    return r


def egm_models(model='Love'):
    """
    Electrogeometric models' parameters. 
    
    Electrogeometric model's parameters for the lightning attachment
    to distribution/ transmission lines.

    Arguments
    ---------
    model: string
        Electrogeometric (EGM) model name from one of the following 
        options: 
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', 'TD',
        where AW stands for Armstrong & Whitehead while BW means 
        Brown & Whitehead and TD is IEEE 1992 T&D Committee model.

    Returns
    -------
    Ac: float
        Parameter A of the EGM for the phase conductor.
    bc: float
        Parameter b of the EGM for the phase conductor.
    Ag: float
        Parameter A of the EGM for the ground.
    bg: float
        Parameter b of the EGM for the ground.

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
    elif model == 'TD':
        # IEEE 1992 T&D Committee
        Ag = 10.
        bg = 0.65
        Ac = 10.
        bc = 0.65
    else:
        raise NotImplementedError(
            'Model {} is not recognized.'.format(model))
    
    return Ac, bc, Ag, bg


def egm(I, model='Love'):
    """
    Electrogeometric model.

    Parameters
    ----------
    I: float
        Lightning current amplitude in kA.
    model: string
        Electrogeometric (EGM) model name from one of the following 
        options: 
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', 'TD',
        where AW stands for Armstrong & Whitehead, while BW means 
        Brown & Whitehead and TD is IEEE 1992 T&D Committee model.

    Returns
    -------
    rg: float
        Striking distance to ground (m),
    rc: float
        Striking distance to phase conductor (m).
    """
    # Obtain EGM model parameters.
    Ac, bc, Ag, bg = egm_models(model)
    # Strike distance to ground.
    rg = egm_distance(I, Ag, bg)
    # Strike distance to phase conductor.
    rc = egm_distance(I, Ac, bc)

    return rg, rc


def max_shielding_current(I, h, y, sg, model='Love'):
    """
    Compute maximum shielding current of the overhead line.

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
        Electrogeometric (EGM) model name from one of the following 
        options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', 'TD',
        where AW stands for Armstrong & Whitehead, while BW means 
        Brown & Whitehead and TD is IEEE 1992 T&D Committee model.

    Returns
    -------
    Igm: float
        Maximum shielding current of the line (kA).
    
    Notes
    -----
    Max. shielding current is determined for the horizontal confi-
    guration of the phase conductors and two shield wires. Certain
    assumptions regarding line geometry have been introduced; see
    references [1,2].
    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    # EGM model parameters.
    Ac, bc, Ag, bg = egm_models(model)
    # EGM Striking distances.
    rg, rc = egm(1., model)

    a = sg/2.
    # Compute max. shielding current value.
    if model in ['AW', 'BW', 'Wagner']:
        # Traditional approach w/o corrections.
        gamma = rc/rg
        alpha = np.arctan(a/(h-y))
        rgm = ((h+y)/2.) / (1. - gamma*np.sin(alpha))
        Igm = (rgm/Ag)**(1./bg)
    
    elif model == 'Young':
        # Young's approach.
        if h >= 18.:
            F = 444./(462. - h)
        else:
            F = 1.
        Agg = F*Ag
        gamma = Ac/Agg
        alpha = np.arctan(a/(h-y))
        rgm = ((h + y)/2.) / (1. - gamma*np.sin(alpha))
        Igm = (rgm/Agg)**(1./bg)
    
    elif model == 'TD':
        # IEEE 1992 T&D Committee model.
        if h > 40.:
            height = 40.
        else:
            height = h
        F = 0.360 + 0.170*np.log(43. - height)
        Agg = F*Ag
        gamma = Ac/Agg
        alpha = np.arctan(a/(h-y))
        rgm = ((h + y)/2.) / (1. - gamma*np.sin(alpha))
        Igm = (rgm/Agg)**(1./bg)
    
    else:
        # Default approach (from Hileman).
        gamma = rc/rg
        alpha = np.arctan(a/(h-y))
        ko = 1. - gamma**2*np.sin(alpha)**2
        rgm = ((h+y)/(2.*ko))*(1. + np.sqrt(1. - ko*(1. + (a/(h+y))**2)))
        Igm = (rgm/Ag)**(1./bg)

    if Igm < 0:
        raise ValueError(f'Maximum shielding current {Igm} is negative.')
    
    return Igm


def exposure_distances(I, h, y, sg, model='Love'):
    """
    Exposure distances to shield wire(s) and phase conductors.

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
        Electrogeometric (EGM) model name from one of the following 
        options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', 'TD',
        where AW stands for Armstrong & Whitehead, while BW means 
        Brown & Whitehead and TD is IEEE 1992 T&D Committee model.

    Returns
    -------
    Dg: float
        Exposure distance to shield wire(s) or ground surface (m).
    Dc: float
        Exposure distance to phase conductor (m).

    Raises
    ------
    ValueError
        If the eight of the phase conductor exceeds that of the 
        shield wire(s).
    
    Notes
    -----
    Exposure distances are determined for the horizontal configu-
    ration of the phase conductors and two shield wires. Certain
    assumptions regarding line geometry have been introduced; see
    references [1,2].
    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    # EGM model.
    rg, rc = egm(I, model)
    # Compute max. shielding current value from the EGM model.
    Igm = max_shielding_current(I, h, y, sg, model)

    if I > Igm:
        Dc = 0.
        thetac = np.arcsin((rg-h)/rc)
        Dg = rc*np.cos(thetac)
    else:
        if rg <= y:
            theta = 0.
        else:
            theta = np.arcsin((rg-y)/rc)
        a = sg/2.
        alpha = np.arctan(a/(h-y))
        beta = np.arcsin(np.sqrt(a**2 + (h-y)**2)/(2.*rc))
        if np.isnan(beta):
            print(f'Parameter "beta": {beta}')
            raise Exception('Failed at computing the "rc" value.')
        Dc = rc*(np.cos(theta) - np.cos(alpha + beta))
        Dg = rc*np.cos(alpha - beta)

    return Dg, Dc


def striking_point(x0, I, h, y, sg, model='Love', shield=True):
    """
    Determine the striking point of the lightning flash.

    Lightning striking point in relation to the overhead line
    is determined from the EGM striking distances and geometry.

    Arguments
    ---------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] 
        in (m) from the distribution line.
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    model: string
        Electrogeometric (EGM) model name from one of the following 
        options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', 'TD',
        where AW stands for Armstrong & Whitehead, while BW means 
        Brown & Whitehead and TD is IEEE 1992 T&D Committee model.
    shield: bool
        Presence of shield wire (True/False).

    Returns
    -------
    strike: int
        Strike point code value (0 to 4).

    Notes
    -----
    Strike event is coded as follows:
        0 - direct stroke to shield wire
        1 - direct stroke to phase conductor (shielding failure)
        2 - indirect (near-by) stroke
        3 - direct stroke to phase conductor without shield wires
        4 - indirect (near-by) stroke without shield wires

    Raises
    ------
    NotImplementedError
    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    if shield:
        # Shield wire is present at the transmission line.
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
        # There is NO shield wire.
        rg, rc = egm(I, model)
        Dc = np.sqrt(rc**2 - (rg-y)**2)
        
        if x0 <= sg/2. + Dc:
            strike = 3  # stroke to phase conductor without shield wires
        elif x0 > sg/2. + Dc:
            strike = 4  # indirect (near-by) stroke without shield wires
        else:
            # Impossible situation encountered.
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


def tower_impedance(height, radius, model='conical'):
    """
    Wave impedance of the transmission line tower.

    Parameters
    ----------
    height: float
        Height oof the tower structure (m).
    radius: float
        Equivalent radius of the tower's base (m).
    model: str
        Model used for computing the wave impedance. Following
        options are provided: `cylindrical`, `conical`.
    
    Returns
    -------
    Zt: float
        Wave impedance of the tower structure (Ohm).
    """
    if model == 'cylindrical':
        # Cylindrical tower shape.
        Zt = 60.*(np.log(2.*np.sqrt(2) * (height/radius)) - 1.)
    
    elif model == 'conical':
        # Conical tower shape.
        Zt = 60.*np.log(np.sqrt(2) * np.sqrt((height/radius)**2 + 1.))
    
    elif model == 'waist':
        # Waisted tower shape.
        theta = np.arctan(radius/height)/2.
        arg = 1./np.tan(theta)
        Zt = np.sqrt(np.pi/4)*60.*(np.log(arg) - np.log(np.sqrt(2)))
    
    else:
        raise NotImplementedError(f'Model name: {model} is not recognized!')
    
    return Zt


def grounding_design_coefficients(grounding_type, length_type, depth):
    """
    Transmission line tower's grounding design coefficients.

    Tower's grounding design is standardized and these are the 
    design coefficients used for computing the low-frequency
    grounding resistance.

    Parameters
    ----------
    grounding_type: str
        Design type (ring, star, or their combination) of the 
        standardized TL tower's grounding system.
    length_type: str
        Design type designation which defines the length of the 
        ring side and/or arm of the star for the standardized
        TL tower's grounding system.
    depth: float, default=0.5
        Depth of burial of the tower's grounding system (it can be
        either 0.5 m or 0.75 m).

    Returns
    -------
    cr: float
        Design coefficient of the grounding system.
    """
    import pandas as pd

    column_names = [
        'depth','1&5','2&10','4&15','5&20','6&25','8&30','10&35','12&40'
    ]
    index_row = [
        'P','P','2P','2P','2xL','2xL','3xL','3xL','4xL','4xL',
        'P+2xL','P+2xL','P+3xL','P+3xL','P+4xL','P+4xL',
        '2P+2xL','2P+2xL','2P+3xL','2P+3xL','2P+4xL','2P+4xL'
    ]
    design_coefficients = [
        0.50, 31.11, 19.57, 11.44, 9.54, 8.21, 6.46, 5.36, 4.59,
        0.75, 27.18, 18.13, 10.91, 9.14, 7.89, 6.24, 5.18, 4.44,
        0.50, 12.61, 9.95, 7.02, 6.14, 5.47, 4.51, 3.85, 3.37,
        0.75, 11.77, 9.47, 6.77, 5.94, 5.30, 4.38, 3.74, 3.28,
        0.50, 14.59, 8.36, 5.99, 4.72, 3.92, 3.36, 2.95, 2.63,
        0.75, 14.02, 8.05, 5.79, 4.56, 3.79, 3.25, 2.86, 2.55,
        0.50, 11.52, 6.52, 4.64, 3.64, 3.00, 2.57, 2.25, 2.00,
        0.75, 11.03, 6.30, 4.49, 3.52, 2.92, 2.50, 2.19, 1.95,
        0.50, 9.84, 5.52, 3.90, 3.05, 2.51, 2.14, 1.87, 1.66,
        0.75, 9.41, 5.33, 3.79, 2.96, 2.44, 2.08, 1.82, 1.62,
        0.50, 6.49, 5.16, 4.27, 6.34, 3.17, 2.82, 2.53, 2.31,
        0.75, 6.25, 4.99, 4.13, 3.52, 3.08, 2.73, 2.46, 2.24,
        0.50, 5.95, 4.50, 3.60, 3.01, 2.59, 2.27, 2.03, 1.84,
        0.75, 5.73, 4.35, 3.50, 2.92, 2.52, 2.21, 1.98, 1.80,
        0.50, 5.50, 4.01, 3.15, 2.60, 2.22, 1.94, 1.72, 1.55,
        0.75, 5.31, 3.89, 3.07, 2.53, 2.16, 1.89, 1.68, 1.51,
        0.50, 4.92, 4.26, 3.70, 3.25, 2.89, 2.61, 2.37, 2.18,
        0.75, 4.75, 4.12, 3.58, 3.15, 2.81, 2.53, 2.30, 2.11,
        0.50, 4.70, 3.90, 3.27, 2.80, 2.45, 2.17, 1.95, 1.77,
        0.75, 4.54, 3.77, 3.17, 2.72, 2.38, 2.11, 1.90, 1.73,
        0.50, 4.51, 3.60, 2.95, 2.48, 2.14, 1.88, 1.68, 1.52,
        0.75, 4.35, 3.49, 2.86, 2.41, 2.06, 1.83, 1.64, 1.48
    ]
    # Construct pandas DataFrame and return a coefficient value.
    design_coefficients = np.array(design_coefficients, dtype=float)
    design_coefficients = np.reshape(design_coefficients, (22, 9))
    coeffs = pd.DataFrame(data=design_coefficients, columns=column_names,
                         index=index_row, dtype=float)
    coeffs.index.name = 'Type'
    coeffs = pd.pivot_table(coeffs, index=['Type', 'depth'])
    cr = coeffs.loc[(grounding_type, depth), length_type]

    return cr


def tower_grounding(grounding_type, length_type, depth=0.5, rho=100.):
    """
    Tower grounding impedance at low-frequency currents.
    
    Computing the impedance (resistance) of the transmission line's
    tower at low-frequency currents, from the standardized types of
    grounding systems used for transmission line towers (i.e. ring
    and star-shape electrode arrangements). Grounding resistance is
    computed from the following relation:

        R0 = rho * cr/100

    where 'cr' is the coefficient of grounding, which is defined for
    each standardized design type of the tower's grounding systems.
    These are ring (P) and star (L) types. Ring electrode is buried
    at the depth of 0.5 m or 0.75 m, at 1 m distance around the tower
    base. Length of the one side of this square ring can be any of
    the following values: 1 m, 2 m, 4 m, 5 m, 6 m, 8 m, 10 m and 12 m.
    Star electrode shape can have 2, 3 or 4 arms, each with the same 
    length, which can be any of the following values: 5 m, 10 m, 15 m,
    20 m, 25 m, 30 m, 35 m and 40 m. Star electrodes are also buried 
    at depths of 0.5 m or 0.75 m. Rings and stars can be further com-
    bined for creating more complex grounding systems.

    Parameters
    ----------
    grounding_type: str
        Design type of the tower's grounding system. Following types
        are recognized:
        - P: ring-type electrode,
        - 2P: double ring electrode,
        - 2xL: star-type electrode with two arms (each extending by
               the same length),
        - 3xL: star-type electrode with three equal-length arms,
        - 4xL: star type electrode with four equal-length arms,
        - P+2xL: ring plus two-pointed star,
        - P+3xL: ring plus three-pointed star,
        - P+4xL: ring plus four pointed star,
        - 2P+2xL: double ring plus two-pointed star,
        - 2P+3xL: double ring plus three-pointed star,
        - 2P+4xL: double ring plus four-pointed star.
    length_type: str
        Design type designation which defines the length of the ring
        side and/or arm of the star. Following values are allowed:
        - 1&5: ring side of 1 m and/or star's arm length of 5 m,
        - 2&10: ring side of 2 m and/or star's arm length of 10 m,
        - 4&15: ring side of 4 m and/or star's arm length of 15 m,
        - 5&20: ring side of 5 m and/or star's arm length of 20 m,
        - 6&25: ring side of 6 m and/or star's arm length of 25 m,
        - 8&30: ring side of 8 m and/or star's arm length of 30 m,
        - 10&35: ring side of 10 m and/or star's arm length of 35 m,
        - 12&40: ring side of 12 m and/or star's arm length of 40 m.
    depth: float, default=0.5
        Depth of burial of the tower's grounding system (it can be
        either 0.5 m or 0.75 m).
    rho: float, default=100
        Average value of the soil resistivity (Ohm*m).
    
    Returns
    -------
    R0: float
        Resistance of the tower's grounding system.
    """
    allowed_grounding_types = [
        'P', '2P', 
        '2xL', '3xL', '4xL', 
        'P+2xL', 'P+3xL', 'P+4xL',
        '2P+2xL', '2P+3xL', '2P+4xL'
    ]
    allowed_length_types = [
        '1&5', '2&10', '4&15', '5&20', 
        '6&25', '8&30', '10&35', '12&40'
    ]
    if grounding_type not in allowed_grounding_types:
        raise Exception(f'Grounding type: {grounding_type} is not recognized!')
    
    if length_type not in allowed_length_types:
        raise Exception(f'Length type: {length_type} is not recognized!')
    
    if depth not in [0.5, 0.75]:
        raise Exception(
            f'Depth value of: {depth} is not allowed. Only 0.5 m and 0.75 m '
            'values are allowed for this parameter.')
    
    # Import grounding design coefficient.
    cr = grounding_design_coefficients(grounding_type, length_type, depth)
    # Compute grounding resistance.
    R0 = (cr/100.) * rho
    
    return R0


def soil_ionization(I, grounding_type, length_type, rho=100., 
                    Eo=400., **kwargs):
    """
    Soil ionization of TL tower's compact grounding systems.

    Simple analysis of the soil ionization of the transmission
    line tower's compact grounding systems.

    Parameters
    ----------
    I: float
        Lightning current amplitude (kA).
    grounding_type: str
        Design type (ring, star, or their combination) of the 
        standardized TL tower's grounding system.
    length_type: str
        Design type designation which defines the length of the 
        ring side and/or arm of the star for the standardized
        TL tower's grounding system.
    rho: float, default=100
        Average value of the soil resistivity (Ohm*m).
    Eo: float, default=400
        Critical value of the electric field necessary for the 
        onset of the soil ionization (kV/m).
    **kwargs: dict
        Additional keyword parameters supplied to the `tower_grounding`
        function.
    
    Returns
    -------
    Ri: float
        Impulse resistance value of the compact grounding system
        with soil ionization accounted for (Ohm).
    """
    R0 = tower_grounding(grounding_type, length_type, **kwargs)
    Ig = (1./(2*np.pi)) * (rho*Eo)/R0**2
    Ri = R0 * np.sqrt(Ig/I)

    return Ri


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
        Wave impedances of the shield wire, including the mirror 
        image in (ohm).

    Raises
    ------
    ValueError
        Height of the phase conductor should not exceed that of the 
        shield wire(s).
    
    Notes
    -----
    Horizontal configuration of the phase conductors with two shield
    wires. Certain assumptions regarding the line geometry have been 
    introduced; see references [1,2].

    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    # Wave impedance.
    Zsw = impedance(h, rad_s)
    a = sg / 2.
    Zswc = 60.*np.log(np.sqrt(a**2 + (h+y)**2) / np.sqrt(a**2 + (h-y)**2))

    return Zsw, Zswc


def indirect_stroke_rusck(x0, I, y, v):
    """
    Indirect stroke (transmission line without shield wire).
    
    Overvoltage is computed according to the Rusck's model.

    Arguments
    ---------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] 
        in (m) from the distribution line.
    I: float
        Lightning current amplitude in kA.
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    v: float
        Velocity of the lightning return stroke (m/us).

    Returns
    -------
    Vc: float
        Overvoltage amplitude (kV).
    """
    c = 300. # m/us
    k = ((30.*I*y)/x0)  # Zo
    Vc = k*(1. + (1./np.sqrt(2.)) * (v/c)*(1. / np.sqrt(1. - 0.5*(v/c)**2)))

    return Vc


def critical_current(x0, y, h, shield, sg, v, CFO, k_cfo=1.,
                     rad_s=2.5e-3, R=10., EPS=1e-2):
    """
    Critical current for indirect stroke flashover.

    Computing critical lightning current amplitude for the
    onset of the flashover event, following indirect nearby
    lighting strikes. Rusck's model is used, in accordance
    with the IEEE Std. 1410. A bisection search algorithm is
    employed for finding the critical current value in each
    case. Coupling factor accounts for the screening effect
    if the shield wire is present.

    Parameters
    ----------
     x0: float
        Perpendicular distance of the lightning strike, (m)
        from the distribution line.
    y: float
        Height of the phase conductor (m).
    h: float
        Height of the shield wire (m).
    shield: bool
        Presence of shield wire (True/False).
    sg: float
        Separation distance between the shield wires (m).
    v: float
        Lightning return stroke velocity (m/us).
    CFO: float
        Critical flashover voltage level of the insulation (kV).
    k_cfo: float, default=1
        Coefficient for correcting the CFO value.
    rad_s: float, default=2.5e-3
        Radius of the shield wire (m).
    R: float, default=10
        Grounding resistance of the shield wire (Ohm).
    EPS: float, default=1e-2
        Tolerance for stopping the bisection search method.

    Returns
    -------
    Icrit: float
        Critical lightning-current amplitude for the onset of
        flashover, (kA).
    """
    # Wave impedances of phase cond. and shield wire.
    Zsw, Zswc = impedances(h, y, sg, rad_s)
    # Coupling factor.
    pr = 1. - (h/y) * (Zswc / (Zsw + 2.*R))
    if pr < 0 or pr > 1:
        raise Exception(f'Coupling factor of {pr} is invalid!')
    
    # Bisection search algorithm.
    Imin, Imax = 0., 500.
    i = 0
    while i < 10_000:
        Imid = (Imin + Imax)/2.
        # Rusck's model per IEEE Std. 1410.
        Vc = indirect_stroke_rusck(x0, Imid, y, v)
        
        if shield:
            # Shield wire provides screening.
            Vc = pr * Vc
        
        if Vc >= k_cfo*CFO:
            # Flashover
            Imax = Imid
        else:
            # No flashover
            Imin = Imid
        
        # Test for convergence.
        if abs(Imax - Imin) <= EPS:
            break
        
        i += 1
    else:
        raise Exception('Iterations did not converge!')

    # Critical current in the final step.
    Icrit = (Imax + Imin)/2.

    return Icrit


def critical_current_chowdhuri(x0, tf, y, h, shield, sg, CFO, 
                               k_cfo=1., rad_s=2.5e-3, R=10., 
                               EPS=1e-2, h_cloud=3000., W=300., 
                               x=0., jakubowski=False):
    """
    Critical current for indirect stroke flashover.

    Computing critical lightning current amplitude for the
    onset of the flashover event, following indirect nearby
    lighting strikes. A Chowdhuri-Gross model is used, with
    a fixed value of the lightning-current wavefront time.
    A bisection search algorithm is employed for finding the
    critical current value in each case. Coupling factor 
    accounts for the screening effect if the shield wire is
    present.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike, (m)
        from the distribution line.
    tf: float
        Lightning-current wavefront time (us).
    y: float
        Height of the phase conductor (m).
    h: float
        Height of the shield wire (m).
    shield: bool
        Presence of shield wire (True/False).
    sg: float
        Separation distance between the shield wires (m).
    CFO: float
        Critical flashover voltage level of the insulation (kV).
    k_cfo: float, default=1
        Coefficient for correcting the CFO value.
    rad_s: float, default=2.5e-3
        Radius of the shield wire (m).
    R: float, default=10
        Grounding resistance of the shield wire (Ohm).
    EPS: float, default=1e-2
        Tolerance for stopping the bisection search method.
    h_cloud: float, default=3000
        Cloud base height (m). Default (3 km) is a typical value.
    W: float, default=300
        Lightning return stroke velocity (m/us). Default value 
        (300 m/us) is according to Chowdhuri. Alternativs are: 
        - 200 m/us (Wagner),
        - 500 m/us (Rusck).
    x: float
        Distance on the line where the overvoltage will be 
        computed (m). Distance of x = 0 is the location on 
        the line that is exactly perpendicular to the lightning 
        strike.
    jakubowski: bool
        Indicator True/False for including the so-called 
        Jakubowski modification to the original Chowdhuri-Gross 
        model. Default state is without the modification.

    Returns
    -------
    Icrit: float
        Critical lightning-current amplitude for the onset of
        flashover, (kA).
    """
    # Wave impedances of phase cond. and shield wire.
    Zsw, Zswc = impedances(h, y, sg, rad_s)
    # Coupling factor.
    pr = 1. - (h/y) * (Zswc / (Zsw + 2.*R))
    if pr < 0 or pr > 1:
        raise Exception(f'Coupling factor of {pr} is invalid!')
    
    # Additional keyword arguments.
    kwargs = {
        'h_cloud': h_cloud,
        'W': W,
        'x': x,
        'jakubowski': jakubowski
    }
    # Bisection search algorithm.
    Imin, Imax = 0., 500.
    i = 0
    while i < 10_000:
        Imid = (Imin + Imax)/2.
        # Chowdhuri-Gross model of indirect strike.
        Vc, r1, r2 = indirect_chowdhuri_gross(x0, Imid, y, tf, **kwargs)
        
        if shield:
            # Shield wire provides screening.
            Vc = pr * Vc
        
        if Vc >= k_cfo*CFO:
            # Flashover
            Imax = Imid
        else:
            # No flashover
            Imin = Imid
        
        # Test for convergence.
        if abs(Imax - Imin) <= EPS:
            break
        
        i += 1
    else:
        raise Exception('Iterations did not converge!')

    # Critical current in the final step.
    Icrit = (Imax + Imin)/2.

    return Icrit


def critical_current_fit(x, y):
    """
    Polynomial fit of the critical current values.

    A polynomial fit of the form: 
        y = a + b*x + c*x**2 + d*x**3
    is used, in the least-squares sence, for fitting
    the (x,y) data of distances and critical lightning
    currents. Function invokes `linalg.lstsq` from the
    `numpy` library.

    Arguments
    ---------
    x: 1d-array
        Array of distances (x-axis values).
    y: 1d-array
        Array of critical currents (y-axis values).
    
    Returns
    -------
    coeffs: 1d-array
        Coefficients of the polinomial fit [a, b, c, d].
    """
    from scipy import linalg

    # Prepare the coefficients matrix.
    X = np.c_[np.ones_like(x), x, x**2, x**3]

    # Solve the least-squares problem.
    coeffs, resid, rank, s = linalg.lstsq(X, y)

    return coeffs


def poly(x, clp):
    """
    Polinomial from the fitted coefficients.

    Polinomial approximation to the CLP curve from
    the coefficients computed from the least-squares.

    Arguments
    ---------
    x: array
        An 1d array holding the x-values data.
    clp: array-like or tuple
        Coefficients of the polinomial, ordered
        from the lowest to the highest exponent.
    
    Returns
    -------
    y: array
        Polinomial values computed at x values.
    """
    y = clp[0] + clp[1]*x + clp[2]*x**2 + clp[3]*x**3
    
    return y


def indirect_chowdhuri_gross(x0, I, y, tf, h_cloud=3000., W=300., x=0.,
                             jakubowski=False):
    """
    Chowdhuri-Gross model. 
    
    Chowdhuri-Gross model of the nearby indirect lightning strike to
    the distribution line (that has no shield wire), which includes
    the so-called Cornfield correction. Jakubowski modification can
    be included in the model as well.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] 
        in (m) from the distribution line.
    I: float
        Lightning current amplitude (kA).
    y: float
        Height of the phase conductor (m).
    tf: float
        Lightning current wavetime front duration (us).
    h_cloud: float
        Cloud base height (m). Default (3 km) is the typical value.
    W: float
        Lightning return stroke velocity (m/us). Default value 
        (300 m/us) is according to Chowdhuri. Alternative values are: 
        - 200 m/us (Wagner),
        - 500 m/us (Rusck).
    x: float
        Distance on the line where the overvoltage will be computed (m).
        Distance of x = 0 is the location on the line that is exactly 
        perpendicular to the lightning strike.
    jakubowski: bool
        Indicator True/False for including the so-called Jakubowski
        modification to the original Chowdhuri-Gross model. Default 
        state is without the modification.

    Returns
    -------
    Vmax: float
        Peak value of the overvoltage at the selected location (x) on 
        the line.
    V, ti: 1d-arrays
        Overvoltage values and associated time instances, respectively.
    """
    # Convert for computation.
    I = I * 1e3     # kA => A
    tf = tf * 1e-6  # us => s
    # Additional data (fixed values).
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

        f9 = b0 * (beta**2*x**2 + x0**2) \
            +beta**2*c**2*t**2 * (1. + beta**2)
        f10 = (2.*beta**2*c*t) \
            *np.sqrt(beta**2*c**2*t**2 + b0*(x**2 + x0**2))
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

        f9a = b0*(beta**2*x**2 + x0**2) \
            + (beta**2*c**2*ttf**2) * (1. + beta**2)
        f10a = (2.*beta**2*c*ttf) \
            *np.sqrt(beta**2*c**2*ttf**2 + b0*(x**2 + x0**2))
        f11a = (c**2*ttf**2 - x**2) / x0**2
        f12a = (f9a - f10a) / (b0**2*x0**2)
        f13a = (f1a*f3a*f5a*f7a) / (f2a*f4a*f6a*f8a)
        
        if jakubowski:
            # Including the Jakubowski modification.
            f14 = (b0*(x**2 + x0**2)) / (beta**2*c**2)
            f15 = (t + np.sqrt(t**2 + f14)) / (t0 + np.sqrt(t0**2 + f14))
            f15a = (ttf + np.sqrt(ttf**2 + f14)) / (t0 + np.sqrt(t0**2 + f14))
            
            if (t < t0):
                V1 = 0.
            else:
                FF1 = f0 * (np.log(f12) 
                            - np.log(f11) 
                            + 0.5*np.log(f13) 
                            + (2.*beta)*np.log(f15))
                V1 = FF1            
            if (t < t0f):
                V2 = 0.
            else:
                FF2 = -f0 * (np.log(f12a) 
                             - np.log(f11a) 
                             + 0.5*np.log(f13a) 
                             + (2.*beta)*np.log(f15a))
                V2 = FF2
        else:
            # Without the Jakubowski modification. 
            if (t < t0):
                V1 = 0.
            else:
                FF1 = f0 * (b0*np.log(f12) 
                            - b0*np.log(f11) 
                            + 0.5*np.log(f13))
                V1 = FF1
            if (t < t0f):
                V2 = 0.
            else:
                FF2 = -f0 * (b0*np.log(f12a) 
                             - b0*np.log(f11a) 
                             + 0.5*np.log(f13a))
                V2 = FF2

        V[i] = (V1 + V2) * 1e-3  # kV
        ti[i] = t*1e6  # us
        t = t + dt

    # Max. absolute value.
    Volt = abs(V)
    Vmax = max(Volt)

    return Vmax, ti, V


def indirect_liew_mar(x0, I, y, tf, h_cloud=3000., W=300., x=0.):
    """
    Liew-Mar model. 
    
    Liew-Mar of the nearby indirect lightning strike to the
    distribution line that has no shield wire.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] 
        in (m) from the distribution line.
    I: float
        Lightning current amplitude (kA).
    y: float
        Height of the phase conductor (m).
    tf: float
        Lightning current wavetime front duration (us).
    h_cloud: float
        Cloud base height (m). Default (3 km) is the typical value.
    W: float
        Lightning return stroke velocity (m/us). Default value 
        (300 m/us) is according to Chowdhuri. Alternative values are: 
        - 200 m/us (Wagner),
        - 500 m/us (Rusck).
    x: float
        Distance on the line where the overvoltage will be computed (m).
        Distance of x = 0 is the location on the line that is exactly 
        perpendicular to the lightning strike.

    Returns
    -------
    Vmax: float
        Peak value of the overvoltage at the selected location (x) 
        on the line.
    V, ti: 1d-arrays
        Overvoltage values and associated time instances, respectively.
    """

    def induced_voltage(x, t):
        """ Induced voltage from the indirect strike.
        x, t: position on the line and time instance
        return: voltage value
        """
        r = np.sqrt(x**2 + x0**2)
        t0 = r / c
        w = 1. / (c*t + x)**2
        wo = 1. / (c*t0 + x)**2
        v = (c*t + x)**2
        vo = (c*t0 + x)**2
        u = 1. / (c*t - x)**2
        uo = 1. / (c*t0 - x)**2
        z = (c*t - x)**2
        zo = (c*t0 - x)**2
        p = (x0**2 + 2.*h_cloud**2) / (x0**4)
        q = 1. / x0**2

        K1 = (30.*I*y) / (tf*beta*c)

        b0 = 1. - beta**2
        b1 = 1. + beta**2
        bt = beta**2 * c**2 * t**2

        P1 = (b0*(beta**2*x**2 + x0**2) + bt*b1) / (b0**2*x0**2)
        P2 = ((2.*beta**2*c*t) 
              * np.sqrt(bt + b0*(x**2 + x0**2))) / (b0**2*x0**2)
        PP1 = K1 * np.log(P1 - P2)

        K2 = (60.*I*y) / (tf*c)

        P3 = np.arcsinh((beta*c*t)/(r * np.sqrt(b0)))
        P4 = np.arcsinh((beta*c*t0)/(r * np.sqrt(b0)))
        PP2 = K2 * (P3 - P4)

        # Function "G"
        P5 = - np.log((c**2*t**2 - x**2) / (x0**2))

        G1 = np.arccosh((u + p)/np.sqrt(p**2 - q**2))
        G2 = np.arccosh((uo + p)/np.sqrt(p**2 - q**2))
        G3 = np.arccosh((z + p/q**2)/np.sqrt(p**2/q**4 - 1./q**2))
        G4 = np.arccosh((zo + p/q**2)/np.sqrt(p**2/q**4 - 1./q**2))
        G5 = np.arccosh((w + p)/np.sqrt(p**2 - q**2))
        G6 = np.arccosh((wo + p)/np.sqrt(p**2 - q**2))
        G7 = np.arccosh((v + p/q**2)/np.sqrt(p**2/q**4 - 1./q**2))
        G8 = np.arccosh((vo + p/q**2)/np.sqrt(p**2/q**4 - 1./q**2))
        GG = K1 * (P5 + 0.5 * (G1 - G2 + G3 - G4 + G5 - G6 + G7 - G8))

        Vi = PP1 + PP2 + GG

        return Vi

    # Convert for computation.
    I = I * 1e3     # kA => A
    tf = tf * 1e-6  # us => s
    # Additional data (fixed values).
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
        r = np.sqrt(x**2 + x0**2)
        t0 = r / c
        zo = beta*c*t0
        tt = t0 + np.sqrt(zo**2 + r**2)/c
        ttf = t - tf

        if (t <= t0):
            V[i] = 0.
        elif ((t > t0) and (t <= t0+tf)):
            V[i] = induced_voltage(x, t)
        elif (t > t0+tf):
            V[i] = induced_voltage(x, t) - induced_voltage(x, ttf)
        
        V[i] = V[i] * 1e-3  # kV
        ti[i] = t * 1e6  # us
        t = t + dt

    # Max. absolute value.
    Volt = abs(V)
    Vmax = max(Volt)

    return Vmax, ti, V


def indirect_shield_wire_absent(
        x0, I, tf, y, v, model_indirect, **kwargs):
    """
    Indirect stroke with TL shield wire absent from the tower.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] 
        in (m) from the distribution line.
    I: float
        Lightning current amplitude in kA.
    tf: float
        Lightning current wave-front time (us).
    y: float
        Height of the phase conductor (m).
    v: float
        Velocity of the lightning return stroke (m/us).
    model_indirect: str
        Model used for computing the indirect strike w/o the
        shield wire. Following three options have been
        implemented:
        - `rusk`: Rusck's model
        - `chow`: Chowdhuri-Gross model
        - `liew`: Liew-Mar model
    **kwargs: dict
        Additional keyword arguments that are forwarded to
        the called function.

    Returns
    -------
    Volt: float
        Overvoltage amplitude (kV).

    Raises
    ------
    NotImplementedError
    """
    if model_indirect == 'rusk':
        # Rusk's model.
        Vc = indirect_stroke_rusck(x0, I, y, v)
    
    elif model_indirect == 'chow':
        # Chowdhuri-Gross model.
        Vc, _, _ = indirect_chowdhuri_gross(x0, I, y, tf, **kwargs)
    
    elif model_indirect == 'liew':
        # Liew-Mar model.
        Vc, _, _ = indirect_liew_mar(x0, I, tf, y, tf, **kwargs)
    
    else:
        raise NotImplementedError(
            f'Model: {model_indirect} is not recognized!')
    
    return Vc


def indirect_shield_wire_present(x0, I, tf, h, y, sg, v, R, rad_s, 
                                 model_indirect, **kwargs):
    """
    Indirect stroke with TL shield wire present on the tower.

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] 
        in (m) from the distribution line.
    I: float
        Lightning current amplitude in kA.
    tf: float
        Lightning current wave-front time (us).
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    v: float
        Velocity of the lightning return stroke (m/us).
    R: float
        Grounding resistance of the shield wire (Ohm).
    rad_s: float
        Radius of the shield wire (m).
    model_indirect: str
        Model used for computing the indirect strike w/o the shield 
        wire. Following three options have been implemented:
        - `rusk`: Rusk's model
        - `chow`: Chowdhuri-Gross model
        - `liew`: Liew-Mar model
    **kwargs: dict
        Additional keyword arguments that are forwarded to the called 
        function.

    Returns
    -------
    Volt: float
        Overvoltage amplitude (kV).

    Raises
    ------
    ValueError, NotImplementedError

    Notes
    -----
    Horizontal configuration of the phase conductors and two shield
    wires. Certain assumptions regarding line geometry have been
    introduced; see references [1,2].
    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    # Overvoltage due to indirect strike w/o shield wire.
    Vc = indirect_shield_wire_absent(x0, I, tf, y, v, model_indirect, **kwargs)

    # Wave impedances of phase cond. and shield wire.
    Zsw, Zswc = impedances(h, y, sg, rad_s)
    # Coupling factor.
    pr = 1. - (h/y) * (Zswc / (Zsw + 2.*R))

    # Overvoltage due to indirect strike with shield wire.
    Volt = pr * Vc

    return Volt


def backflashover_hileman(I, h, y, sg, Ri, rad_s, span=150.):
    """
    Backflashover analysis.

    Lightning strike to shield wire and the backflashover overvoltage.
    The backflashover computation is according to the simplified IEEE 
    method.

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
    Ri: float
        Stricken tower's impulse grounding resistance (Ohm).
    rad_s: float
        Radius of the shield wire (m).
    span: float
        Span length (default value: 150 m on distribution lines).

    Returns
    -------
    Volt: float
        Overvoltage amplitude (kV).

    Raises
    ------
    ValueError
        Height of the phase conductor should not exceed that of the 
        shield wire(s).
    
    Notes
    -----
    Analysis of the backflashover phenomenon is presented in detail 
    in the following reference:
        A. R. Hileman, Insulation Coordination for Power Systems, 
        CRC Press, Boca Raton (FL), 1999, pp. 373-423.
    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    c = 300. # m/us
    Zsw, Zswc = impedances(h, y, sg, rad_s)
    Z = Zsw/2.
    Rn = Ri/2.  # Grounding resistance of the tower at the other side of the
    # span it is assumed here that it equals half of the (impulse) grounding
    # impedance of the stricken transmission line tower; this assumption does
    # not have significant impact on the outcome.
    Zw = ((2.*Ri**2*Z) / (Z + Ri)**2) * ((Z - Rn) / (Z + Rn))
    Zi = (Ri*Z) / (Z + Ri)
    Fi = ((Z - Ri) / (Z + Ri)) * ((Z - Rn) / (Z + Rn))
    Tau = span / c
    T = 2.  # fixed; see the original paper for more info.
    N = int(np.floor(T / (2.*Tau)))
    kv = I * Tau * Zw * ((1. - Fi**N) / (1. - Fi)**2 
                         -(float(N)*Fi**N) / (1. - Fi))
    Vt = 0.5 * I * T * (Zi - (Zw * (1. - Fi**N)) / (1. - Fi)) + kv
    CF = Zswc / Zsw

    # Overvoltage
    Volt = Vt * (1. - CF)

    return Volt


def backflashover_cigre(I, Un, R0, rho, h, y, rad_s, span, 
                        r_tower, CFO=150., KPF=0.7, C=0.35, 
                        Eo=400., tower_model='conical', 
                        eps_Ri=0.1, eps_tf=0.01):
    """
    CIGRE method for computing backflashover overvoltage. 
    
    This is the so-called CIGRE method for the backflashover 
    computation on overhead transmission (and/or distribution) 
    lines.

    Parameters
    ----------
    I: float
        Lightning current amplitude (kA).
    Un: float
        Nominal voltage of the line (kV).
    R0: float
        Grounding resistance (at low-current level) of the distri-
        bution line's tower (Ohm). Typical values range between 10 
        to 50 Ohms.
    rho: float
        Average soil resistivity at the tower's site (Ohm/m). 
        Parameters `rho` and `R0` define the factor `rho/R0` which
        is typically found in the range between 10 and 50.    
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor, (m).
    rad_s: float
        Shield wire radius (m).
    span: float
        Average length of a single span of the distribution line (m).
    r_tower: float
        Equivalent radius of the transmission tower base, (m).
    CFO: float
        Critical flashover voltage level of the line's insulation (kV).
    KPF: float
        Dimensionless factor (so-called power frequency factor) for
        correcting the nominal voltage of the line. For a horizontal
        configuration of the line the recommended value is 0.7, while
        for a vertical configuration the recommended value is 0.4.
    C: float
        Coupling factor (dimensionless) between the shield wire(s) 
        and the phase conductors. Recommended average value of this 
        factor is 0.35.
    Eo: float
        Electric field strength for the inception of the soil ioni-
        zation at the tower grounding (kV/m). Recommended value is 
        400 kV/m.
    tower_model: str
        Model used for computing the wave impedance of the tower 
        structure. Following options are provided: `cylindrical`, 
        `conical`.
    eps_Ri: float
        Tolerance of the tower's grounding impulse impedance value 
        for terminating the iterative computation procedure of the 
        CIGRE method.
    eps_tf: float
        Tolerance of the lightning current wave-front time value for
        terminating the iterative computation procedure of the CIGRE
        method.

    Returns
    -------
    tf: float
        Wave-front time of the lightning current causing back-
        flashover (us).
    Ic: float
        Critical lightning current amplitude for the back-
        flashover event (kA).
    Vc: float
        Backflashover overvoltage amplitude (kV).

    Raises
    ------
    Exception
        General exception is raised if the outer while loop 
        exhausts the counter without convergence.

    Notes
    -----
    A. R. Hileman, Insulation Coordination for Power Systems, 
        CRC Press, Boca Raton (FL), 1999, pp. 373-423.
    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    c = 300. # light-speed (m/us)
    Ta = y/c
    Tt = h/c
    Ts = span/c

    # Wave impedance of the tower.
    Zt = tower_impedance(h, r_tower, tower_model)
    # Surge impedance of the shield wire.
    Zg = impedance(h, rad_s)
    # Voltage correction factor.
    VPF = KPF * Un * np.sqrt(2.)/np.sqrt(3.)

    i = 0
    tf = 2.5
    while i < 10_000:
        Ri = R0/2.
        while True:
            Re = (Ri * Zg)/(Zg + 2.*Ri)
            alphaT = (Zt - Ri)/(Zt + Ri)
            alphaR = Zg/(Zg + 2.*Ri)
            Tau = (Zg/Ri)*Ts
            DV = (alphaT*Zt*(Ta - C*Tt))/(tf*Re*(1. - C))
            CFOns = (0.997 + 2.82/Tau)*(1. + DV)*(1. - 0.2*(1. + DV)*VPF/CFO)\
                    * (1. - 0.09*(1. + 10./Tau)*DV)*np.exp(-DV*tf/13.)*CFO
            Ktt = Re + alphaT*Zt*(Tt/tf)
            Kta = Re + alphaT*Zt*(Ta/tf)
            
            if 2.*(Ts/tf) < 1.:
                k1 = 1.-2.*(Ts/tf)
            else:
                k1 = 0.
            if 4.*(Ts/tf) < 1.:
                k2 = 1. - 4.*(Ts/tf)
            else:
                k2 = 0.
            if 6.*(Ts/tf) < 1.:
                k3 = 1. - 6.*(Ts/tf)
            else:
                k3 = 0.
            if 8.*(Ts/tf) < 1.:
                k4 = 1. - 8.*(Ts/tf)
            else:
                k4 = 0.
            
            Ksp = 1. - alphaR*(1. - alphaT)*(k1 
                + alphaR*alphaT*k2 
                + (alphaR*alphaT)**2*k3 
                + (alphaR*alphaT)**3*k4)
            Ic = (CFOns - VPF)/(Ksp*(Kta - C*Ktt))
            IR = (Re/Ri)*Ic
            Ig = (rho*Eo)/(2. * np.pi * R0**2)
            Rin = R0/np.sqrt(1. + IR/Ig)

            # Test for convergence
            if abs(Rin - Ri) <= eps_Ri:
                break
            else:
                Ri = Rin
        else:
            raise Exception('Error: Iterative method did not converge!')
        
        tfn = 0.207*Ic**0.53
        # Test for convergence
        if abs(tfn - tf) <= eps_tf:
            break
        else:
            tf = tfn

        i += 1
    else:
        raise Exception('Error: Iterative method did not converge!')
    
    # Backflashover overvoltage.
    Vc = (1. - C) * Re * I

    return tf, Ic, Vc


def backflashover_cigre_simple(I, Un, R0, rho, h, rad_s, span=150., 
                               CFO=150., KPF=0.7, C=0.35, Eo=400., 
                               eps_Ri=0.1):
    """
    Simplified CIGRE method for computing backflashover overvoltage.
    
    This is the so-called simplified CIGRE method for calculating 
    backflashover on overhead transmission (and/or distribution) 
    lines.

    Parameters
    ----------
    I: float
        Lightning current amplitude (kA).
    Un: float
        Nominal voltage of the line (kV).
    R0: float
        Grounding resistance (at low-current level) of the distri-
        bution line's tower (Ohm). Typical values range between 
        10 to 50 Ohms.
    rho: float
        Average soil resistivity at the tower's site (Ohm/m). Para-
        meters `rho` and `R0` define the factor `rho/R0` which is 
        typically found in the range between 10 and 50.
    CFO: float
        Critical flashover voltage level of the line's insulation 
        (kV).
    KPF: float
        Dimensionless factor (so-called power frequency factor) for
        correcting the nominal voltage of the line. For a horizontal
        configuration of the line the recommended value is 0.7, 
        while for a vertical configuration the recommended value is 
        0.4.
    span: float
        Average length of a single span of the distribution line (m).
    h: float
        Height of the shield wire (m).
    rad_s: float
        Shield wire radius (m).
    C: float
        Coupling factor (dimensionless) between the shield wire(s) 
        and the phase conductors. Recommended average value of this 
        factor is 0.35.
    Eo: float
        Electric field strength for the inception of the soil ioni-
        zation at the tower grounding (kV/m). Recommended value is 
        400 kV/m.
    eps_Ri: float
        Tolerance of the tower's grounding impulse impedance value 
        for terminating the iterative computation procedure of the 
        simplified method.
    
    Returns
    -------
    Ic: float
        Critical current value for the onset of the backflashover 
        event, (kA).
    Vc: float
        Backflashover overvoltage value across the insulator strings 
        on the stricken tower, (kV).
    
    Raises
    ------
    Exception
        General exception is raised if the while loop exhausts the 
        counter without convergence.
    
    Notes
    -----
    This method is described in detail (including its underlying 
    assumptions and limitations) in the following reference:
        A. R. Hileman, Insulation Coordination for Power Systems, 
        CRC Press, Boca Raton (FL), 1999, pp. 373-423.
    """
    # Power frequency phase voltage.
    VPF = KPF * (Un*np.sqrt(2.)/np.sqrt(3.))
    # Travel time of the single span (us)
    c = 300. # m/us
    Ts = span / c 
    # Surge impedance of the shield wire(s).
    Zg = impedance(h, rad_s)
    
    k = 0
    Ri = R0/2.
    while k < 10_000:
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
    
    # Backflashover overvoltage.
    Vc = (1. - C) * Re * I

    return Ic, Vc


def backflashover(Un, I, h, y, sg, R0, Ri, rad_s, r_tower,
                  span=150., CFO=150., KPF=0.7, C=0.35, rho=100., 
                  Eo=400., tower_model='conical', model_bfr='hileman', 
                  eps_Ri=0.1, eps_tf=0.01):
    """
    Backflashover computation.
    
    Computing overvoltage amplitude from the backflashover incident, 
    by means of any of the three different methods that have been 
    provided.

    Parameters
    ----------
    Un: float
        Nominal voltage of the line (kV).
    I: float
        Lightning current amplitude (kA).
    h: float
        Height of the shield wire (m).
    y: float
        Height of the phase conductor (m).
    sg: float
        Separation distance between the shield wires (m).
    R0: float or None
        Grounding resistance (at low-current level) of the distribution
        line's tower (Ohm).
    Ri: float or None
        Tower's (impulse) resistance/ impedance (Ohm), with or without
        the soil ionization.
    rad_s: float
        Shield wire radius (m).
    r_tower: float
        Equivalent radius of the transmission tower base, (m).
    span:float
        Average length of a single span of the distribution line (m).
    CFO: float
        Critical flashover voltage level of the line's insulation (kV).
    KPF: float
        Dimensionless factor (so-called power frequency factor) for
        correcting the nominal voltage of the line. For a horizontal
        configuration of the line the recommended value is 0.7, while
        for a vertical configuration the recommended value is 0.4.
    C: float
        Coupling factor (dimensionless) between the shield wire(s) and 
        the phase conductors. Recommended average value of this factor 
        is 0.35.
    rho: float
        Average soil resistivity at the tower's site (Ohm/m). Para-
        meters `rho` and `R0` define the factor `rho/R0` which is 
        typically found in the range between 10 and 50.    
    Eo: float
        Electric field strength for the inception of the soil ioni-
        zation at the tower grounding (kV/m). Recommended value is 
        400 kV/m.
    tower_model: str
        Model used for computing the wave impedance of the tower 
        structure. Following options are provided: `cylindrical`, 
        `conical`.
    model_bfr: str
        Model used for computing the backflashover overvoltage. 
        Following three options have been implemented:
        - `hileman`: Hileman's model
        - `cigre`: CIGRE model
        - `cigre-simple`: Simplified CIGRE model
    eps_Ri: float
        Tolerance of the tower's grounding impulse impedance value 
        for terminating the iterative computation procedure of the 
        CIGRE method.
    eps_tf: float
        Tolerance of the lightning current wave-front time value 
        for terminating the iterative computation procedure of the 
        CIGRE method.

    Returns
    -------
    Vc: float
        Overvoltage amplitude as a consequence of the backflashover
        incident (kV).
    
    Notes
    -----
    If the values `R0` or `Ri` are `None` then it is assumed that the
    TL tower's grounding system is standardized and concentrated. It 
    is a square ring with 1 m length sides, buried at the 0.5 m depth.
    """
    # Tower's concentrated grounding system.
    if R0 is None:
        # Fall back to the default standardized grounding system.
        grounding_type = 'P'  # square ring type
        length_type = '1&5'   # 1 m side length
        # Low-frequency resistance
        R0 = tower_grounding(grounding_type, length_type, rho=rho)
    
    if Ri is None:
        # Fall back to the default standardized grounding system.
        grounding_type = 'P'  # square ring type
        length_type = '1&5'   # 1 m side length
        # Soil ionization (simplified)
        Ri = soil_ionization(I, grounding_type, length_type, rho=rho, Eo=Eo)

    # Select model for the backflashover analysis.
    if model_bfr == 'hileman':
        # Hileman's model of backflashover analysis.
        Vc = backflashover_hileman(I, h, y, sg, Ri, rad_s, span)
    
    elif model_bfr == 'cigre':
        # CIGRE model of backflashover analysis.
        params = backflashover_cigre(
            I, Un, R0, rho, h, y, rad_s, span, r_tower, CFO, 
            KPF, C, Eo, tower_model, eps_Ri, eps_tf)
        Vc = params[2]
    
    elif model_bfr == 'cigre-simple':
        # Simplified CIGRE model of backflashover analysis.
        params = backflashover_cigre_simple(
            I, Un, R0, rho, h, rad_s, span, 
            CFO, KPF, C, Eo, eps_Ri)
        Vc = params[1]
    
    else:
        raise NotImplementedError(
            f'Backflashover model: {model_bfr} is not recognized!')

    return Vc


def compute_overvoltage(x0, I, tf, h, y, sg, w, Ri, rad_c, rad_s, R, 
                        model='Love', shield=True, span=150.,
                        **kwargs):
    """
    Compute overvoltage amplitude.

    Compute overvoltage amplitude on transmission line for different
    types of lightning strokes, where the strike event has been coded
    as follows:
        0 - direct stroke to shield wire (i.e. backflashover)
        1 - direct stroke to phase conductor (shielding failure)
        2 - indirect (near-by) stroke where shield wires are present
        3 - direct stroke to phase conductor without shield wires
        4 - indirect (near-by) stroke where shield wires are absent

    Parameters
    ----------
    x0: float
        Perpendicular distance of the lightning strike [0, xmax] 
        in (m) from the distribution line.
    I: float
        Lightning current amplitude in kA.
    tf: float
        Lightning current wave-front time (us).
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
    model: string
        Electrogeometric (EGM) model name from one of the following 
        options: 
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', 'TD', 
        where AW stands for Armstrong & Whitehead, while BW means 
        Brown & Whitehead and TD is IEEE 1992 T&D Committee model.
    shield: bool
        Presence of shield wire (True/False).
    span: float
        Span length (default value: 150 m on distribution lines).
    **kwargs: dict
        Dictionary holding additional keyword arguments that are 
        forwarded to other functions that are called from here.

    Returns
    -------
    stroke: int
        Type of stroke (0 - 4) that was considered.
    V: float
        Overvoltage amplitude from the stroke incident (kV).
    
    Notes
    -----
    Overvoltage ampltide depends on the type of lightning strike,
    which, in-turn, depends on the presence or absence of shield 
    wires. Lightning strike type, at the same time, depends on the 
    striking point, which is determined by the exposure distances
    (from the line geometry, strike location, and the EGM model).
    """
    from operator import itemgetter

    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    # Unpacking extra arguments.
    (Un, R0, rho, CFO, KPF, C, Eo, r_tower,
        tower_model, model_bfr, model_bfr_random,
        model_indirect, eps_Ri, eps_tf
        ) = itemgetter(
            'Un', 'R0', 'rho', 'CFO', 'KPF', 'C', 'Eo', 'r_tower',
            'tower_model', 'model_bfr', 'model_bfr_random',
            'model_indirect', 'eps_Ri', 'eps_tf'
        )(kwargs)

    c = 300. # speed of light in free space (m/us)
    # Compute return-stroke velocity.
    v = c/np.sqrt(1. + w/I)
    
    # Point of lightning strike.
    stroke = striking_point(x0, I, h, y, sg, model, shield)

    # Compute overvoltage on the transmission line.
    if shield:
        # Shield wire is present on the transmission line.
        if stroke == 1:
            # Stroke to phase conductor (shielding failure).
            V = phase_conductor(I, y, rad_c)
        elif stroke == 2:
            # Indirect stroke.
            V = indirect_shield_wire_present(x0, I, tf, h, y, sg, v, R, rad_s, 
                                             model_indirect)
        elif stroke == 0:
            # Stroke to shield wire (backflashover).
            if model_bfr_random:
                # Randomize backflashover model selection.
                model_bfr = np.random.choice(['hileman', 'cigre-simple'])
            # Compute backflashover.
            V = backflashover(Un, I, h, y, sg, R0, Ri, rad_s, r_tower,
                              span, CFO, KPF, C, rho, Eo, tower_model, 
                              model_bfr, eps_Ri, eps_tf)
        else:
            # Impossible situation encountered.
            raise NotImplementedError('Impossible situation encountered!')
    
    else:
        # There is NO shield wire on the transmission line.
        if stroke == 3:
            # Stroke to phase conductor.
            V = phase_conductor(I, y, rad_c)
        elif stroke == 4:
            # Indirect stroke.
            V = indirect_shield_wire_absent(x0, I, tf, y, v, model_indirect)
        else:
            # Impossible situation encountered.
            raise NotImplementedError('Impossible situation encountered!')
    
    return stroke, V


def transmission_line(N, h, y, sg, distances, amplitudes, fronts, 
                      w, Ri, egm_models, shield_wire, near_models, 
                      Un, R0, rho, r_tower, tower_model='conical', 
                      CFO=150., k_cfo=1., rad_c=5e-3, rad_s=2.5e-3, 
                      R=10., span=150., KPF=0.7, C=0.35, Eo=400., 
                      model_bfr='hileman', model_bfr_random=False,
                      eps_Ri=0.1, eps_tf=0.01):
    """
    Flashover analysis on overhead electric power line.

    Determine if the flashover has occurred or not for any overhead 
    line with horizontal configuration of phase conductors and two
    shield wires.

    Parameters
    ----------
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
    fronts: array
        Lightning current wave-front times (us).
    w: array
        Lightning return stroke velocity (m/us).
    Ri: array
        Tower's (impulse) resistance/ impedance (Ohm).
    egm_model: list of strings
        Electrogeometric (EGM) model name from one of the following 
        options:
        'Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson', 'TD',
        where AW stands for Armstrong & Whitehead, while BW means 
        Brown & Whitehead and TD is IEEE 1992 T&D Committee model.
    shield_wire: list of bools
        Presence of shield wire (True/False).
    near_models: list of strings
        Models used for computing the indirect strike.
    Un: float
        Nominal voltage of the transmission line (kV).
    R0: float or None
        Grounding resistance (at low-current level) of the distri-
        bution line's tower (Ohm). Typical values range between 10 
        to 50 Ohms.
    rho: float
        Average soil resistivity at the tower's site (Ohm/m). Para-
        meters `rho` and `R0` define the factor `rho/R0` which is 
        typically found in the range between 10 and 50.
    r_tower: float
        Equivalent radius of the transmission tower base, (m).
    tower_model: str
        Model used for computing the wave impedance of the tower 
        structure. Following options are provided: `cylindrical`, 
        `conical`.
    CFO: float
        Critical flashover voltage level (kV).
    k_cfo: float
        Factor for increasing the CFO level due to the lightning 
        wave-front duration/steepness (i.e. CFO of the insulation 
        increases for short duration wave-fronts).
    rad_c: float
        Phase conductor radius (m).
    rad_s: float
        Shield wire radius (m).
    R: float
        Grounding resistance of shield wire (Ohm).
    span: float
        Span length (default value: 150 m on distribution lines).
    KPF: float
        Dimensionless factor (so-called power frequency factor) for
        correcting the nominal voltage of the line. For a horizontal
        configuration of the line the recommended value is 0.7, while
        for a vertical configuration the recommended value is 0.4.
    C: float
        Coupling factor (dimensionless) between the shield wire(s) 
        and the phase conductors. Recommended average value of this 
        factor is 0.35.
    Eo: float
        Electric field strength for the inception of the soil ioni-
        zation at the tower grounding (kV/m). Recommended value is 
        400 kV/m.
    model_bfr: str
        Model used for computing the backflashover overvoltage. 
        Following three options have been implemented:
        - `hileman`: Hileman's model
        - `cigre`: CIGRE model
        - `cigre-simple`: Simplified CIGRE model
    model_bfr_random: bool
        True/False indicator for randomizing selection of the model
        for computing the backflashover event.
    eps_Ri: float
        Tolerance of the tower's grounding impulse impedance value 
        for terminating the iterative computation procedure of the 
        CIGREmethod.
    eps_tf: float
        Tolerance of the lightning current wave-front time value for
        terminating the iterative computation procedure of the CIGRE
        method.
 
    Returns
    -------
    flash: array of bools 
        Flashover (0/1) indicator: 1 - flashover, 0 - no flashover.

    Notes
    -----
    Typical distribution line geometry is used for default values.
    Line has horizontal configuration of phase conductors and two
    shield wires.
    """
    if y > h:
        raise ValueError(
            'y > h: Height of the phase cond. (y) should NOT exceed'
            ' that of the shield wire (h).')
    
    # Extra parameters.
    kwargs = {
        'Un': Un, 'R0': R0, 'CFO': CFO, 'KPF': KPF,'C': C, 'Eo': Eo,
        'r_tower': r_tower, 'tower_model': tower_model, 'rho': rho,
        'model_bfr': model_bfr, 'model_bfr_random': model_bfr_random,
        'eps_Ri': eps_Ri, 'eps_tf': eps_tf
        }

    flash = np.empty_like(amplitudes)
    # Flashover computation for each lightning strike.
    for j in range(N):
        # Compute overvoltage value.
        _, overvoltage = compute_overvoltage(
            distances[j], amplitudes[j], fronts[j], h, y, sg, w[j],
            Ri[j], rad_c, rad_s, R, egm_models[j], shield_wire[j],
            span, model_indirect=near_models[j], **kwargs)
        
        # Determine if there was a flashover or not.
        if abs(overvoltage) > k_cfo*CFO:
            flashover = True
        else:
            flashover = False
        flash[j] = flashover

    return flash


def generate_samples(N, XMAX=500, RiTL=50., muI=31.1, sigmaI=0.484,
                     muTf=3.83, sigmaTf=0.55, rhoxy=0.47, joint=True):
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
    muI: float
        Median value of lightning current amplitudes statistical
        distribution (kA).
    sigmaI: float
        Standard deviation of lightning current amplitudes statistical
        distribution. Default values for `muI` and `sigmaI` are taken 
        from the IEC 62305 and IEC 60071.
    muTf: float
        Median value of lightning current wave-fron times statistical
        distribution (us).
    sigmaTf: float
        Standard deviation of lightning current wave-front times 
        statistical distribution. Default values are taken from the 
        CIGRE and IEEE WGs.
    rhoxy: float
        Coefficient of statistical correlation between lightning-
        current amplitudes and wave-front times.
    joint: bool
        Indicator True/False which determines if the amplitudes and 
        wave-front times of the lightning currents will be treated as
        dependent (True) or independent (False) statistical variables.

    Returns
    -------
    I, tf, w, distances, Ri: 1d-arrays of floats
        Random samples from the appropriate statistical distributions, 
        respectivly of: amplitudes, wave-front times, return stroke 
        velocities, stroke distances, tower and grounding impulse 
        impedances,
    shield wire: 1d-array of bools
        Random samples from the Bernoulli distribution of shield wire 
        presence/absence indicators,
    egm_models: 1d-array of str
        Random sample of EGM model types,
    near_models: 1d-array or str
        Random sample of indirect strike analysis model types.
    """
    from scipy import stats
    from lightning import copula_gauss_bivariate

    if joint:
        # Lightning current amplitudes and wave-front times are 
        # statistically dependent random variables. Random data
        # is generated using the bivariate Gaussian Copula approach.
        u, v = copula_gauss_bivariate(N, rhoxy)
        tf = stats.lognorm.ppf(u, sigmaTf, scale=muTf)
        I = stats.lognorm.ppf(v, sigmaI, scale=muI)
    else:
        # Lightning current amplitudes and wave-front times are 
        # statistically independent random variables. Each is 
        # generated from the Log-Normal distribution.
        # Lightning current amplitudes (IEC 62305)
        tf = stats.lognorm(s=sigmaTf, loc=0., scale=muTf).rvs(size=N)
        I = stats.lognorm(s=sigmaI, loc=0., scale=muI).rvs(size=N)
    
    # Return stroke velocity is computed from the following formula:
    # v = c/sqrt(1 + w/I), where "c" is the speed of light in free
    # space and "I" is the lightning-current ampltude. Parameter "w"
    # has fixed uniform distribution: U[50, 500].
    w = np.random.uniform(low=50., high=500., size=N)

    # Distance of lightning stroke from the transmission line.
    # Uniform distribution: U[0, XMAX].
    distances = np.random.uniform(low=0., high=XMAX, size=N)
    
    # Tower grounding impulse resistance.
    # Normal distribution truncated on the left side at zero.
    sigma_frac = 0.2
    lower_limit = -1/sigma_frac
    Ri = stats.truncnorm(a=lower_limit, b=np.Inf, 
                         loc=RiTL, scale=sigma_frac*RiTL).rvs(size=N)
    
    # Presence or absence of the shield wire(s).
    # Bernoulli distribution where shield wire is present in 50% of cases.
    shield_wire = stats.bernoulli(p=0.5).rvs(size=N)

    # Select EGM models according to the custom probability levels.
    egm_models_all = ['Wagner', 'Young', 'AW', 'BW', 'Love', 'Anderson']
    probabilities = [0.1, 0.2, 0.1, 0.1, 0.3, 0.2]  # custom levels (sum=1)
    egm_models = np.random.choice(egm_models_all, size=N, replace=True,
                                  p=probabilities)

    # Select models for indirect lightning analysis with custom probability.
    near_models_list = ['rusk', 'chow', 'liew']
    # In the interest of the backwards compatibility with previous results,
    # 90% of dataset for the indirect strikes comes from the Rusck's model.
    probas_list = [0.9, 0.05, 0.05]  # custom levels (sum=1)
    near_models = np.random.choice(near_models_list, size=N, replace=True,
                                   p=probas_list)
    
    return I, tf, w, distances, Ri, shield_wire, egm_models, near_models


def generate_dataset(N, h, y, sg, cfo, *args, XMAX=500., RiTL=50., 
                     muI=31.1, sigmaI=0.484, muTf=3.83, sigmaTf=0.55, 
                     rhoxy=0.47, joint=True, export=False, **kwargs):
    """
    Generate a random dataset.

    Generating a random dataset of lightning flashovers on medium 
    voltage distribution lines, by means of the Monte Carlo simu-
    lation of different possible flashover incidents (from both 
    direct and indirect lightning strikes).

    Parameters
    ----------
    N: int
        Number of simulations for each distribution line geometry.
    h: array
        Array of shield wire heights, one entry for each distri-
        bution line, in meters.
    y: array
        Array of phase conductor heights, one entry for each 
        distribution line, in meters.
    sg: array
        Array of distance values between shield wires, one entry 
        for each distribution line, in meters.
    cfo: array
        Array of critical flashover voltage values, one entry 
        for each distribution line, in kV.
    XMAX: float
        Maximum perpendicular distance from the distribution line
        for which lightning flashover interaction is considered 
        feasible, in meters.
    RiTL: float
        Average value of the tower's grounding impedance (used as 
        a mean value in the appropriate Normal distribution), in Ohm.
    muI: float
        Median value of lightning current amplitudes statistical 
        distribution, in kA.
    sigmaI: float
        Standard deviation of lightning current amplitudes statis-
        tical distribution. 
    muTf: float
        Median value of lightning current wave-fron times statis-
        tical distribution (us).
    sigmaTf: float
        Standard deviation of lightning current wave-front times 
        statistical distribution.
    rhoxy: float
        Coefficient of statistical correlation between lightning-
        current amplitudes and wave-front times.
    joint: bool
        Indicator True/False which determines if the amplitudes and 
        wave-front times of the lightning currents will be treated 
        as dependent (True) or independent (False) statistical 
        variables.
    export: bool
        Indicator True/False for exporting generated dataset into
        the CSV format.
    *args: list
        Additional positional parameters (Un, R0, rho, r_tower) 
        that will be forwarded to the `transmission_line` function.
    **kwargs: dict
        Additional keyword parameters that will be forwarded to the 
        `transmission_line` function. Many of these can be left with 
        their default values.

    Returns
    -------
    data: pd.DataFrame
        Randomly generated dataset of lightning flashovers on distri-
        bution lines.
    
    Notes
    -----
    Dataset considers several distribution lines at the same time, 
    where each line is of the same type, but with a different geometry. 
    Each line has a flat configuration of phase conductors (at the 
    height `y`), and with a double shield wires (at the height `h`) 
    that are seprated by distance `sg`. Each line can also have a 
    different critical flashover voltage (CFO) value.
    """
    data = {
        'dist': [],   # distances of the lightning strikes
        'ampl': [],   # lightning-current amplitudes
        'front': [],  # lightning-current wave-front times
        'shield': [], # presence/absence of shield wire(s)
        'veloc': [],  # lightning return-stroke velocities
        'Ri': [],     # impulse impedances of TL tower
        'EGM': [],    # EGM models
        'ind': [],    # indirect stroke models
        'CFO': [],    # CFO values of the TL insulation
        'height': [], # height of the TL phase conductors
        'flash': []   # flashover indicator (1=flashover)
    }

    for j in range(y.size):
        # For each distribution line geometry
        height = np.repeat(y[j], N)
        cfo_value = np.repeat(cfo[j], N)
        
        # Generate random samples
        amps, tf, w, dists, Ri, sws, egms, near_models = generate_samples(
            N, XMAX, RiTL, muI, sigmaI, muTf, sigmaTf, rhoxy, joint)
                
        # Simulate flashovers
        f = transmission_line(N, h[j], y[j], sg[j], 
                              dists, amps, tf, w, Ri, egms, sws, near_models,
                              *args, CFO=cfo[j], **kwargs)
        
        # Store data as dictionary
        data['dist'].append(dists)
        data['ampl'].append(amps)
        data['front'].append(tf)
        data['shield'].append(sws)
        data['veloc'].append(w)
        data['Ri'].append(Ri)
        data['EGM'].append(egms)
        data['ind'].append(near_models)
        data['CFO'].append(cfo_value)
        data['height'].append(height)
        data['flash'].append(f)

    # Transfer data from dictionary to pandas DataFrame
    dataset = {}
    for key, value in data.items():
        dataset[key] = np.array(value).flatten()
    data = pd.DataFrame(data=dataset)

    if export:
        # Export data to csv file
        data.to_csv('distlines.csv')

    return data


def lightning_current_pdf(x, mu, sigma):
    """
    Lightning current probability distribution.

    Probability density function (PDF) of the Log-Normal statis-
    tical distribution. It can serve for generating random ampli-
    tudes, wave-front times and wave-front steepnesses (with 
    introduction of the appropriate median values and standard 
    deviations).

    Parameters
    ----------
    x: float
        Value of the lightning current parameter at which the
        Log-Normal distribution is to be evaluated.
    mu: float
        Median value of the Log-Normal distribution of lightning
        current. It can be amplitude (kA), wave-front time (us), 
        or wave-front steepness (kA/us).
    sigma: float
        Standard deviation of the Log-Normal distribution of
        lightning current. It can be for the amplitude, wave-front
        time or wave-front steepness.
    
    Returns
    -------
    pdf: float
        Probability density function (PDF) value.
    """
    denominator = (np.sqrt(2.*np.pi)*x*sigma)
    pdf = np.exp(-(np.log(x) - np.log(mu))**2 / (2.*sigma**2)) / denominator
    
    # Convert `nan` to numerical values
    return np.nan_to_num(pdf)


def risk_of_flashover(support, y_hat, method='simpson', muI=31.1, 
                      sigmaI=0.484):
    """
    Risk of flashover.

    Compute the risk of flashover with a numerical integration 
    routine.

    Parameters
    ----------
    support: array
        Sample points at which the function to be integrated 
        is defined. It is advisable to have odd number of sample 
        points.
    y_hat: array
        Function values that are integrated at the support.
    method: string
        Integration method to use: 'simpson' or 'trapezoid'.
    muI: float
        Median value of lightning current amplitudes (kA).
    sigmaI: float
        Standard deviation of lightning current amplitudes.

    Returns
    -------
    risk: float
        Risk of insulation flashover.
    
    Raises
    ------
    NotImplementedError
    
    Note
    ----
    For an odd number of samples that are equally spaced the 
    result od simpson's method is exact if the function is a 
    polynomial of order 3 or less.
    """
    from scipy import integrate

    # Probability density function.
    pdf = lightning_current_pdf(support, muI, sigmaI)
    # Integrand
    product = pdf * y_hat
    integrand = np.nan_to_num(product)
    
    if method == 'simpson':
        # Simpson's rule.
        risk = integrate.simpson(integrand, support)
    elif method == 'trapezoid':
        # Trapezoid rule.
        risk = integrate.trapezoid(integrand, support)
    else:
        raise NotImplementedError('Method {} not recognized!'.format(method))
    
    return risk


def risk_curve_fit(x, a, b):
    """
    Fitting the risk curve.

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


if __name__ == "__main__":
    """Showcase of various aspects of the library."""
    import matplotlib.pyplot as plt

    from utils import jitter

    # Figure style using matplotlib
    fig_style = 'seaborn-v0_8-colorblind'
    plt.style.use(fig_style)

    # Number of random samples
    N = 1000
    
    # Distribution line geometry (single line example):
    Un = 20.  # nominal voltage (kV)
    h = 11.5  # shield wire height (m)
    y = 10.   # phase conductor height (m)
    sg = 3.   # distance between shield wires (m)

    # Overhead line has a horizontal configuration of
    # the phase conductors, with two shield wires.
    # Shielding angle of the line is computed from the
    # outer phase's conductor's arm length, which is 
    # hereafter assumed as sg/2! Any shielding angle can
    #  be taken into account by adjusting the sg value.
    
    # Tower's grounding system
    grounding_type = 'P'  # ring-type
    length_type = '1&5'   # 5 m length
    r_tower = 1.
    rho_soil = 100.
    
    print('Running ...')

    # Generate random samples for the Monte Carlo simulation.
    # The same samples are used for all transmission lines.
    amps, tf, w, dists, Ri, sws, egms, near_models = generate_samples(N)
    
    # Flashover analysis for a single transmission line.
    R0 = tower_grounding(grounding_type, length_type, rho=rho_soil)
    args = (Un, R0, rho_soil, r_tower)
    kwargs= {# user-defined keyword arguments
        'tower_model': 'cylindrical',
        'model_bfr': 'hileman',
        'CFO': 150.,
    }
    fl = transmission_line(N, h, y, sg, dists, amps, tf, w, Ri, egms, sws, 
                           near_models, *args, **kwargs)

    # Graphical visualization of simulation results
    # marginal of distance
    fig, ax = plt.subplots(figsize=(7, 5))
    jitter(ax, dists[sws==True], fl[sws==True], s=20,
           c='darkorange', label='shield wire')
    jitter(ax, dists[sws==False], fl[sws==False], s=5,
           c='royalblue', label='NO shield wire')
    ax.legend(loc='center right')
    ax.set_ylabel('Flashover probability')
    ax.set_xlabel('Distance (m)')
    ax.grid(True)
    plt.show()
    # marginal of amplitude
    fig, ax = plt.subplots(figsize=(7, 5))
    jitter(ax, amps[sws==True], fl[sws==True], s=20,
           c='darkorange', label='shield wire')
    jitter(ax, amps[sws==False], fl[sws==False], s=5,
           c='royalblue', label='NO shield wire')
    ax.legend(loc='center right')
    ax.set_ylabel('Flashover probability')
    ax.set_xlabel('Amplitude (kA)')
    ax.grid(True)
    plt.show()
    # in two dimensions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dists[fl==0], amps[fl==0], s=20,
               color='darkorange', label='NO flashover')
    ax.scatter(dists[fl==1], amps[fl==1], s=20,
               color='royalblue', label='flashover')
    ax.legend(loc='upper right')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Amplitude (kA)')
    ax.grid(True)
    plt.show()

    # Rusk's model (indirect strike w/o shield wires)
    Vmax = indirect_stroke_rusck(100., 10., y, 100.)
    print(f'Vmax = {Vmax:.1f} (kV) Rusck')

    # Chowdhuri-Gross model (indirect strike w/o shield wires)
    fig, ax = plt.subplots(figsize=(6,4))
    fig.suptitle('Chowdhuri-Gross model')
    ax.set_title('Distance: 100 (m), Amplitude: 10 (kA), Front-time: 5 (us)',
                 fontsize=10)
    for distance in [0., 2500.]:
        _, ti, V = indirect_chowdhuri_gross(100., 10., y, 5., x=distance)
        ax.plot(ti, V, ls='-', lw=2, label=f'x = {distance*1e-3:.1f} (km)')
    ax.legend(loc='lower right')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('Overvoltage (kV)')
    ax.set_xlim(0, 50)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    
    # Liew-Mar model (indirect strike w/o shield wires)
    fig, ax = plt.subplots(figsize=(6,4))
    fig.suptitle('Liew-Mar model')
    ax.set_title('Distance: 100 (m), Amplitude: 10 (kA), Front-time: 5 (us)',
                 fontsize=10)
    for distance in [0., 2500.]:
        _, ti, V = indirect_liew_mar(100., 10., y, 5., x=distance)
        ax.plot(ti, V, ls='-', lw=2, label=f'x = {distance*1e-3:.1f} (km)')
    ax.legend(loc='lower right')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('Overvoltage (kV)')
    ax.set_xlim(0, 50)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    
    # Testing: backflashover on HV transmission line
    print('Backflashover:')
    Vc = backflashover_hileman(30., 25., 22., 3., 5., 5e-3, span=350.)
    print(f'Vc = {Vc:>.2f} (kV)', 'Hileman')
    # Testing: backflashover CIGRE method on HV transmission line
    _, Ic, Vc = backflashover_cigre(
        30., 110., 10., 100., 25., 22., rad_s=5e-3, span=350., r_tower=2., 
        CFO=700., KPF=0.4)
    print(f'Vc = {Vc:>.2f} (kV)', 'CIGRE')
    # Testing: backflashover simplified CIGRE method
    Ic, Vc = backflashover_cigre_simple(
        30., 110., 10., 100., 25., rad_s=5e-3, span=350., 
        CFO=700., KPF=0.4)
    print(f'Vc = {Vc:>.2f} (kV)', 'CIGRE simplified')
