import pandas as pd 

class MetricLogger:
    def __init__(self):
        self._init_data()

    def retset(self):
        self._init_data()

    def _init_data(self):
        self._data = {
            'model_name': [],
            'metric': [],
            'metric_value': [],
            'data_index': [], 
            'epoch': []
        }

    def log_row(self, model_name, metric, metric_value, data_index, epoch):
        self._data['model_name'].append(model_name)
        self._data['metric'].append(str(metric))
        self._data['metric_value'].append(float(metric_value))
        self._data['data_index'].append(int(data_index))
        self._data['epoch'].append(int(epoch))

    def to_dataframe(self):
        return pd.DataFrame.from_dict(self._data, orient='index').T
    
    def to_csv(self, file_path):
        self.to_dataframe().to_csv(file_path, index=False)

    def print_mean_last_epoch(self):
        df = self.to_dataframe()
        df = df[df['epoch'] == df['epoch'].max()]
        print(df.groupby(['metric', 'model_name'])['metric_value'].mean())


