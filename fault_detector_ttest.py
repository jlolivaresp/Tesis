def fault_detector_ttest(lista, stand_dv, conf_lev, delta_mean):
    from math import ceil
    import scipy.stats
    import scipy as sp
    import math
    import numpy as np
    N = ceil((stand_dv*sp.stats.norm.ppf((1+conf_lev)/2)/delta_mean)**2)

    fault_counter = 0
    cont = 0
    t_falla = np.array([])
    while N+cont <= len(lista):
        ttest_issermann = (lista[:N+cont].mean()-lista[N+cont:2*N+cont].mean())*math.sqrt(N*N*(2*N -
                                                                           2)/(2*N))/math.sqrt((2*N-2)*(stand_dv**2))
        if abs(ttest_issermann) > sp.stats.t.ppf(conf_lev, 2*N-2):
            t_falla = np.append(t_falla,lista[N+cont:2*N+cont].index[0])
            fault_counter += 1
        cont += 1
    inicio_falla = t_falla[0]
    fin_falla = t_falla[-1]
    return inicio_falla,fin_falla

