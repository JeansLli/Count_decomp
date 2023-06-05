import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from io import BytesIO
import persistable

# fetch the data from the hdbscan repo
url = "https://github.com/scikit-learn-contrib/hdbscan/blob/4052692af994610adc9f72486a47c905dd527c94/notebooks/clusterable_data.npy?raw=true"
f = urlopen(url)
rf = f.read()
data = np.load(BytesIO(rf))

# plot the data
plt.figure(figsize=(10,10))
plt.scatter(data[:,0], data[:,1], alpha=0.5)
plt.show()


p = persistable.Persistable(data, n_neighbors="all")

pi = persistable.PersistableInteractive(p)
port = pi.start_UI()


