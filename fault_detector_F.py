def fault_detector_F(lista,delta_var,conf_lev):
    import numpy as np
    import scipy.stats as sp

    # Calculamos el tamano N de la muestra para detectar un cambio en la varianza delta_var

    if np.std(lista) > 0.001:
        N = 1
        X_2_table = sp.chi2.ppf(q=conf_lev,df=N)
        X_2 = ((lista.std()+delta_var)/np.std(lista))**2

        while X_2 < X_2_table:
            N += 1
            X_2_table = sp.chi2.ppf(q=conf_lev,df=N)
            X_2 = (N - 1)*(((lista.std()+delta_var)/np.std(lista))**2)
            if N > len(lista)/2:
                break

        # Iteramos a lo largo de los datos para comprobar si hubo o no un cambio en la varianza

        fault_counter = 0
        cont = 0
        t_falla = np.array([])
        while N+cont <= len(lista)-N:
            dfn = len(lista[:N+cont])
            if np.std(lista[N+cont:2*N+cont]) > 0.001:
                F_N1_N2 = (np.std(lista[:N+cont])/np.std(lista[N+cont:2*N+cont]))**2
                F_table = sp.f.ppf(q=conf_lev,dfn=dfn,dfd=N)

                if F_N1_N2 > F_table:
                    t_falla = np.append(t_falla,lista[N+cont:2*N+cont].index[0])
                    fault_counter += 1
                cont += 1
            else:
                t_falla = [0,0]
                break
    else:
        t_falla = [0,0]
    inicio_falla = t_falla[0]
    fin_falla = t_falla[-1]
    return inicio_falla,fin_falla