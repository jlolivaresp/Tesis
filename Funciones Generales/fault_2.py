class fault_detector(object):

    def __init__(self, vector_faulty):
        self.Vector_faulty = vector_faulty

    def t_test(self, non_faulty_data, stand_dev, conf_lev, delta_mean, N ='auto'):

        from math import ceil
        import scipy.stats
        import scipy as sp
        import math
        import numpy as np

        vector_to_analyze = np.copy(self.Vector_faulty)

        if N == 'auto':
            N = ceil((stand_dev*sp.stats.norm.ppf((1+conf_lev)/2)/delta_mean)**2)

        falla_bool = np.zeros(len(vector_to_analyze))

        if N <= len(vector_to_analyze):
            cont = 0
            while N+cont <= len(vector_to_analyze):
                ttest_issermann = (non_faulty_data.mean() - vector_to_analyze[cont:N + cont].mean()) * \
                                  math.sqrt(len(non_faulty_data) * N * (len(non_faulty_data) + N - 2) /
                                            (len(non_faulty_data) + N)) / \
                                  math.sqrt((len(non_faulty_data) - 1) * np.std(non_faulty_data) ** 2 + (N - 1) *
                                            stand_dev ** 2)
                if abs(ttest_issermann) > sp.stats.t.ppf(conf_lev, len(non_faulty_data)+N-2):
                    falla_bool[cont:N+cont] = np.ones(N)
                cont += 1
        else:
            print('N debe ser menor a la longitud del vector lista')

        number_of_faults_detected = len(falla_bool[falla_bool == 1])
        vector_detected = vector_to_analyze[falla_bool == 1]

        return vector_detected, falla_bool, number_of_faults_detected, N

    def f_test(self, non_faulty_data, std, delta_var, conf_lev, N ='auto'):

        import scipy.stats
        import scipy as sp
        import numpy as np

        vector_to_analyze = np.copy(self.Vector_faulty)

        falla_bool = np.zeros(len(vector_to_analyze))
        if np.std(vector_to_analyze) > 0.0001:
            if N == 'auto':
                N = 2
                X_2_table = sp.stats.chi2.ppf(q=conf_lev, df=N-1)
                X_2 = ((std+np.sqrt(delta_var))/std)**2
                while X_2 < X_2_table:
                    N += 1
                    X_2_table = sp.stats.chi2.ppf(q=conf_lev, df=N-1)
                    X_2 = (N - 1)*(((std+np.sqrt(delta_var))/std)**2)
                    if N > len(vector_to_analyze):
                        print('No se cuenta con suficientes valores para detectar un '
                              'cambio en la varianza de {}, pruebe indicando el valor de N'.format(delta_var))
                        break
            cont = 0
            while N+cont <= len(vector_to_analyze):
                dfn = len(non_faulty_data)
                if np.std(vector_to_analyze[cont:N+cont]) > 0.0001:
                    F_N1_N2 = (np.std(vector_to_analyze[cont:N+cont])/np.std(non_faulty_data))**2
                    #F_N1_N2 = (max(F)/min(F))**2
                    F_table = sp.stats.f.ppf(q=conf_lev, dfn=dfn - 1, dfd=N - 1)

                    if F_N1_N2 > F_table:
                        falla_bool[cont:N+cont] = np.ones(N)
                cont += 1
        fault_counter = len(falla_bool[falla_bool == 1])
        vector_detected = vector_to_analyze[falla_bool == 1]

        return vector_detected, falla_bool, fault_counter, N

class fault_generator(object):

    def __init__(self, vector_non_faulty):
        self.Vector_non_faulty = vector_non_faulty

    def drift(self, start, stop, step, change):

        import numpy as np

        vector_faulty = np.copy(self.Vector_non_faulty)
        vector_faulty[int(start/step):int(stop/step)] += np.linspace(0, change, (stop-start)/step)
        vector_faulty[int(stop/step):] += change
        slope = change/(stop-start)
        drifted_positions_bool = vector_faulty - self.Vector_non_faulty
        drifted_positions_bool = drifted_positions_bool != 0

        return vector_faulty, slope, drifted_positions_bool

    def variance(self, start, stop, step, stand_dev, random_seed=0):

        import numpy as np

        np.random.seed(random_seed)
        vector_faulty = np.copy(self.Vector_non_faulty)
        vector_faulty[int(start/step):int(stop/step)] += np.random.normal(0, stand_dev, int((stop-start)/step))
        faulty_fraction = (stop-start)/(step*len(vector_faulty))
        varianced_positions_bool = vector_faulty - self.Vector_non_faulty
        varianced_positions_bool = varianced_positions_bool != 0

        return vector_faulty, faulty_fraction, varianced_positions_bool

    def random_pulse(self, start, stop, step, N, amplitude, random_seed=0, mode='random'):

        import numpy as np
        import random

        vector_faulty = np.copy(self.Vector_non_faulty)
        after_fault = vector_faulty[int(start/step):int(stop/step)]
        position_after_fault = np.asarray(list(enumerate(after_fault,start=int(start/step))))

        random.seed(random_seed)
        rand_sample_after_fault = position_after_fault[random.sample(range(0, len(position_after_fault)), N), :]

        if mode == 'random':
            np.random.seed(random_seed)
            pulsed_values = [[i[0], i[1] + abs(np.random.normal(0, amplitude))] for i in rand_sample_after_fault]
            for j in pulsed_values:
                vector_faulty[j[0]] = j[1]
        elif mode == 'fixed':
            pulsed_values = [[i[0], i[1] + amplitude] for i in rand_sample_after_fault]
            for j in pulsed_values:
                vector_faulty[j[0]] = j[1]

        faulty_fraction = N/len(vector_faulty)
        pulsed_positions_bool = vector_faulty - self.Vector_non_faulty
        pulsed_positions_bool = pulsed_positions_bool != 0

        return vector_faulty, faulty_fraction, pulsed_positions_bool