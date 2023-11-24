import plotly.graph_objs as go
import numpy as np

n = 100
x = np.arange(1, n + 1)
fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=0.1 * np.random.randn(n) + np.exp(-x ** 2),
                         mode='lines+markers',
                         name='Loss in training'))

fig.add_trace(go.Scatter(x=x, y=0.1 * np.random.randn(n) + np.exp(-x ** 2),
                         mode='lines+markers',
                         name='Loss in validation'))

fig.update_layout(xaxis_title='Epoch',
                  yaxis_title='ln(MSE)',
                  xaxis=dict(title=dict(font=dict(size=18))),
                  yaxis=dict(title=dict(font=dict(size=18))),
                  width=700,
                  height=600,
                  legend=dict(x=0.7, y=0.9),
                  margin=dict(l=0, r=0, t=0, b=0)
                  )
print('saving')
fig.write_image('prueba.pdf')
