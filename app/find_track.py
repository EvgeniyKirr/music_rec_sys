from os import listdir
from numpy import mean
from hashlib import md5
from model import ResNet
from tools import WavLoader
from warnings import filterwarnings
from clickhouse_connect import get_client
from torch import load, cuda, device as to_device


filterwarnings("ignore")

dir_path = './playlist'

files = listdir(dir_path)
files.remove('.placeholder')

if not files:
    print('Playlist folder is empty')
    exit()

client = get_client(host='clickhouse', username='default')
device = to_device('cuda' if cuda.is_available() else 'cpu')
model = ResNet([3, 4, 6, 3]).to(device)
model.load_state_dict(load('model.pt', map_location=device))
model.stage_inference()

vectors = []
for file in files:
    path = f'{dir_path}/{file}'

    with open(path, 'rb') as f:
        hash_md5 = md5(f.read()).hexdigest()

    query = 'select vector from cached_tracks where hash = {hash:FixedString(32)}'
    result = client.query(query, parameters={'hash':hash_md5}).result_rows

    if result:
        vector = result[0][0]
    else:
        tensor = WavLoader(path).get_tensor().to(device)
        if len(tensor) == 0:
            print(f'Track [{path}] its too short and skipped')
            continue
        vector = model(tensor).cpu().detach().numpy().flatten()

        data, schema = [(hash_md5, vector)], ['hash', 'vector']
        client.insert('cached_tracks', data, column_names=schema)

    vectors.append(vector)

avg_vector = mean(vectors, axis=0)
client.insert('comparison_vectors', [[avg_vector]], column_names=['vector'])

recommended_result = client.query('select artist, name from recommended_track').result_rows
client.insert('penalized_tracks', recommended_result, column_names=['artist', 'name'])

for artist, name in recommended_result:
    print(f'{artist}: {name}')
