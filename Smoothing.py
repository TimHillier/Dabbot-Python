'''
Smoothes pose classification
'''
class EMADictSmoothing(object):
    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha
        self._data_in_window = []
    
    '''
    Smothes given pose classification.
    - Comput Exponential moving average for every pose. 
    '''
    def __call__(self, data):
        # Add data.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get All The Keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get Smoothed Values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update Factor. 
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data



