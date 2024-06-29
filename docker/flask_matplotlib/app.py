import base64
from io import BytesIO

import pandas as pd

from flask import Flask

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
app = Flask(__name__)

@app.route('/')
def hello_world():
   return "Hello world!"

@app.route('/plot')
def plot():
    # Generate the figure **without using pyplot**.
    #fig = Figure()
    #ax = fig.subplots()
    #ax.plot([1, 2])
    ax = df.plot()
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig = plt.gcf()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

if __name__ == "__main__":
   df = pd.read_csv('data.csv')
   df = df.astype(int)
   app.run(host='0.0.0.0', port=5001, debug=True)
