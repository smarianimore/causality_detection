import pandas as pd
import json
import requests
from requests.auth import HTTPBasicAuth
from time import sleep
import numpy as np
from tqdm import tqdm


class iCasa:
    def __init__(self, instance):
        self.__auth = HTTPBasicAuth('reader', 'xai')

        self.__get_device_url = f'https://explainableai.fr/{instance}/icasa/devices/devices'
        self.__get_zone_url = f'https://explainableai.fr/{instance}/icasa/zones/zones'

        self.__put_device_url = f'https://explainableai.fr/{instance}/etienne/device/autonomic/device/'
        self.__put_zone_url = f'https://explainableai.fr/{instance}/etienne/zones/'

        self.__speed_url = f'https://explainableai.fr/icasaSimulation/icasa/clocks/clock/default'
        self.__restart_url = 'https://explainableai.fr/icasaRestart'
        # self.set_devices = {"temperature_controller": "m_target_temperature",
        #                     'window': 'm_open',
        #                     'window': 'm_blinds_open',
        #                     'heater': 'm_heater.powerLevel'}
        #
        # self.set_zones = {'outdoor': 'myTemperature'}

        # self.read_devices = {"heater": ['heater.powerLevel', 'm_heater.powerLevel'],
        #                      "thermometer": ['etienne.temperature'],
        #                      "out_thermometer": ['etienne.temperature'],
        #                      'window': ['blinds_open', 'window.opened', 'm_open', 'm_blinds_open'],
        #                      "temperature_controller": ['m_target_temperature', 'state'],
        #                      }
        #
        # self.controlable = ['window__m_open', 'window__m_blinds_open'] \
        #                    + ["heater__m_heater.powerLevel", "out_thermometer__etienne.temperature",
        #                       'temperature_controller__state', 'temperature_controller__m_target_temperature']
        #
        # self.data = None
        #
        # self.name2kind = {'outdoor': 'zone', 'window': 'device', 'out_thermometer': 'device',
        #                   'heater': 'device', 'temperature_controller': 'device'}
        #
        # self.name = lambda k, p, v: (self.name2kind[k], (k, p), float(v))
        #
        # self.state = [('window', 'm_open'), ('window', 'blinds_open'), ('heater', 'm_heater.powerLevel'),
        #               ('temperature_controller', 'state')]
        # self.temperature = [0, 7, 17, 23, 28, 30]
        # self.reset_auto()
        # self._do([self.name('temperature_controller', 'm_target_temperature', 20)], resp_time=0)

    def acceleration(self):
        resp = requests.get(self.__speed_url, auth=self.__auth)
        if resp.status_code != 200:
            print("acceleration failed!")
            return
        param = resp.json()
        print(param)
        param['factor'] = 100
        requests.put(self.__speed_url,
                     data=json.dumps(param),
                     auth=self.__auth)

        print('acceleration speed up !')

    def fetch_data(self) -> pd.DataFrame:
        new_data = {}

        # Read sensor data
        resp_device = requests.get(self.__get_device_url, auth=self.__auth)
        if resp_device.status_code != 200:
            print("data loading fail!")

        for device in resp_device.json():
            id = device['id']
            if id not in self.read_devices: continue
            for param in device['properties']:
                if param['name'] in self.read_devices[id]:
                    new_data[id + '__' + param['name']] = [param['value']]

        # update with new sensor data
        return pd.DataFrame(new_data)

    def add_record(self) -> pd.DataFrame:
        new_data = self.fetch_data()
        if self.data is None:
            self.data = new_data
        else:
            self.data = pd.concat((self.data, new_data))
        return self.data

    def reset_auto(self):
        T = np.random.choice(self.temperature)
        to_do = [(self.name(remain[0], remain[1], -1)) for remain in self.state] \
                + [('zone', ('outdoor', 'myTemperature'), float(T))]
        self._do(to_do, resp_time=0)

    def _do(self, evidence, resp_time=0, verbose=False) -> pd.DataFrame:
        """
        evidence: list of (kind, name, val)
        resp_time : (in sec) waiting time before fetching data
        """
        for kind, (name, property), value in evidence:
            if kind == 'device':
                resp = requests.put(self.__put_device_url + f'/{name}',
                                    data=json.dumps({'id': name, property: value}),
                                    auth=self.__auth)
            elif kind == 'zone':
                resp = requests.put(self.__put_zone_url + f'/{name}',
                                    data=json.dumps({property: value}),
                                    auth=self.__auth)
            else:
                assert False, 'not recongnize option: kind'

            if resp.status_code != 200:
                print("pushing data fail!")

        print('waiting for response...', end='') if verbose else None
        sleep(resp_time) if resp_time > 0 else None
        print('[OK]') if verbose else None

        return self.fetch_data()

    def do(self, evidence, size=10, seed=12, resp_time=3, verbose=True):  # NB: change to do_evidence for usage with learner??
        np.random.seed(seed)

        if evidence is None:
            evidence = {}

        # reformat evidence
        evidence_ = [self.name(k.split('__')[0], k.split('__')[1], v) for k, v in evidence.items()]

        column = [f'{k}__{v[0]}' for k, v in self.read_devices.items()]
        df = pd.DataFrame({node: [] for node in column}).astype(int)
        for i in range(size):
            df = pd.concat((df, self._do(evidence_, resp_time=resp_time, verbose=True)))
            self.reset_auto()

        df.reset_index(drop=True, inplace=True)
        return df


if __name__ == '__main__':
    home = iCasa('icasaInterface')
    home.acceleration()
    # state = [('window', 'm_open'), ('window', 'm_blinds_open'), ('heater', 'm_heater.powerLevel')]
    #
    # n_data = 1000
    # np.random.seed(0)
    # temperature = [0, 7, 17, 23, 28, 30]
    #
    # for _ in tqdm(range(n_data)):
    #     T = np.random.choice(temperature)
    #     # select random one with random value and set the rest in auto mode
    #     state_ = state.copy()
    #     np.random.shuffle(state_)
    #     select = state_.pop()
    #     val = np.random.choice([0, 1, -1])
    #     to_do = [(home.name(select[0], select[1], val))] \
    #             + [(home.name(remain[0], remain[1], -1)) for remain in state_] \
    #             + [('zone', ('outdoor', 'myTemperature'), float(T))]
    #
    #     resp = home._do(to_do, resp_time=3)
    #
    #     home.add_record().to_csv('ocik/demo/store/icasa2.csv', index=False)
    #
    # home.data.to_csv('ocik/demo/store/icasa2.csv', index=False)
