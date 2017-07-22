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
