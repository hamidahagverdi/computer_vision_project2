import numpy as np

# Returns DeltaE2000 between Lab1 and Lab2 (each shape (...,3))
def deltaE_ciede2000(Lab1, Lab2):
    Lab1 = Lab1.astype(np.float64)
    Lab2 = Lab2.astype(np.float64)

    L1, a1, b1 = Lab1[..., 0], Lab1[..., 1], Lab1[..., 2]
    L2, a2, b2 = Lab2[..., 0], Lab2[..., 1], Lab2[..., 2]

    C1 = np.sqrt(a1*a1 + b1*b1)
    C2 = np.sqrt(a2*a2 + b2*b2)
    C_bar = (C1 + C2) / 2.0

    C_bar7 = C_bar**7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25**7 + 1e-12)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = np.sqrt(a1p*a1p + b1*b1)
    C2p = np.sqrt(a2p*a2p + b2*b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p*C2p) == 0, 0, dhp)

    dHp = 2*np.sqrt(C1p*C2p) * np.sin(np.radians(dhp)/2.0)

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)

    hp_bar = np.where((C1p*C2p) == 0, hsum, 0)
    hp_bar = np.where((C1p*C2p) != 0,
                      np.where(hdiff > 180, (hsum + 360)/2.0, hsum/2.0),
                      hp_bar)
    hp_bar = hp_bar % 360

    T = (1
         - 0.17*np.cos(np.radians(hp_bar - 30))
         + 0.24*np.cos(np.radians(2*hp_bar))
         + 0.32*np.cos(np.radians(3*hp_bar + 6))
         - 0.20*np.cos(np.radians(4*hp_bar - 63)))

    dtheta = 30*np.exp(-((hp_bar - 275)/25)**2)
    Rc = 2*np.sqrt((Cp_bar**7) / (Cp_bar**7 + 25**7 + 1e-12))

    Sl = 1 + (0.015*(Lp_bar - 50)**2) / np.sqrt(20 + (Lp_bar - 50)**2 + 1e-12)
    Sc = 1 + 0.045*Cp_bar
    Sh = 1 + 0.015*Cp_bar*T

    Rt = -np.sin(np.radians(2*dtheta)) * Rc

    Kl = Kc = Kh = 1.0
    dE = np.sqrt(
        (dLp/(Kl*Sl))**2 +
        (dCp/(Kc*Sc))**2 +
        (dHp/(Kh*Sh))**2 +
        Rt*(dCp/(Kc*Sc))*(dHp/(Kh*Sh))
    )
    return dE