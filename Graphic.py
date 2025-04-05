from matplotlib import pyplot as plt
import numpy as np

class Graphic:
    def __init__(self, x: str, y: str, title: str, hyperparameters: dict = None, window_size: int = 10):
        plt.ion()
        self.fig, self.axis = plt.subplots()

        self.axis.set_xlabel(x)
        self.axis.set_ylabel(y)
        self.axis.set_title(title)

        self.x_labels = np.array([], dtype=np.int64)
        self.y_labels = np.array([], dtype=np.float32)

        self.line, = self.axis.plot([], [], color='red', alpha=0.3, label='Really line')
        self.smooth_line, = self.axis.plot([], [], color='red', label='Smooth line')

        self.axis.legend()

        if hyperparameters:
            hyper_text = '\n'.join([f'{key}: {value}' for key, value in zip(hyperparameters.keys(), hyperparameters.values())])
            self.axis.text(0.95, 0.05, hyper_text, verticalalignment='bottom', horizontalalignment='right',
                           transform=self.axis.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        self.window_size = window_size

    def moving_average(self):
        smooth_y = np.zeros_like(self.y_labels)
        for i in range(1, len(self.y_labels) + 1):
            window = min(self.window_size, i)  # Постепенное увеличение окна
            smooth_y[i - 1] = np.mean(self.y_labels[i - window:i])  # Среднее по доступным данным
        return smooth_y
    
    def update(self, x: int, y: float):
        self.x_labels = np.append(self.x_labels, x)
        self.y_labels = np.append(self.y_labels, y)

        self.line.set_data(self.x_labels, self.y_labels)
        
        smooth_y = self.moving_average()
        smooth_x = self.x_labels[max(0, len(self.x_labels) - len(smooth_y)):]
        self.smooth_line.set_data(smooth_x, smooth_y)

        self.axis.relim()
        self.axis.autoscale_view()

        plt.draw()
        plt.pause(0.005)

    def show(self):
        plt.ioff()

        self.line.set_data(self.x_labels, self.y_labels)

        smooth_y = self.moving_average()
        smooth_x = self.x_labels[max(0, len(self.x_labels) - len(smooth_y)):]
        self.smooth_line.set_data(smooth_x, smooth_y)

        self.axis.relim()
        self.axis.autoscale_view()

        plt.draw()
        plt.pause(0.005)

        plt.show()