class fault_generator(object):

    def __init__(self, vector_non_faulty):
        self.Vector_non_faulty = vector_non_faulty

    def drift(self, start, stop, step, change):

        import numpy as np

        vector_faulty = np.copy(self.Vector_non_faulty)
        vector_faulty[start:stop] += np.linspace(0, change, (stop-start)/step)
        vector_faulty[stop:] += change
        faulty_fraction = (stop-start)/(step*len(vector_faulty))
        return vector_faulty, faulty_fraction

    def variance(self, start, stop, step, stand_dev):

        import numpy as np

        vector_faulty = np.copy(self.Vector_non_faulty)
        vector_faulty[start:stop] += np.random.normal(0, stand_dev, int((stop-start)/step))
        faulty_fraction = (stop-start)/(step*len(vector_faulty))
        return vector_faulty, faulty_fraction

    def random_pulse(self, start, stop, N, amplitude, mode = 'random'):

        import numpy as np
        import random

        vector_faulty = np.copy(self.Vector_non_faulty)
        after_fault = vector_faulty[start:stop]
        position_after_fault = np.asarray(list(enumerate(after_fault,start=start)))

        rand_sample_after_fault = position_after_fault[random.sample(range(0, len(position_after_fault)), N), :]

        if mode == 'random':
            pulsed_values = [[i[0], i[1] + abs(np.random.normal(0, amplitude))] for i in rand_sample_after_fault]
            for j in pulsed_values:
                vector_faulty[j[0]] = j[1]
        elif mode == 'fixed':
            pulsed_values = [[i[0], i[1] + amplitude] for i in rand_sample_after_fault]
            for j in pulsed_values:
                vector_faulty[j[0]] = j[1]

        faulty_fraction = N/len(vector_faulty)

        return vector_faulty, faulty_fraction
