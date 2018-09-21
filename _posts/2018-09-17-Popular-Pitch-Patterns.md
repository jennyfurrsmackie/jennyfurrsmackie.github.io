---
published: true
---

## Temporal Pattern Recognition: Predicting the Millenial Whoop

### INTRODUCTION

**PURPOSE:**
    
     To predict song popularity from audio analysis elements. This project will cover an investigation of the hypothetically popular "Millenial Whoop" [MW] Research Questions include:
    
    A) Can we train a model to identify the "Millenial Whoop" in a spotify song? 
    B) Will the occurance of the MW coorelate with a song's popularity metric as determined by Spotify? 
    
    In this project we investigate Questions:

    A) by modelling MW pitch patterns. 
    B) will be investigated in later versions of this project.

    The [Millenial Whoop](https://fr.wikipedia.org/wiki/Millennial_whoop) is a 3-5 chord progression phenomena discovered commonly in modern pop music.  For a quick overview of the MW, see here: [The Millenial Whoop is Taking Over Pop Music](https://www.youtube.com/watch?v=MN23lFKfpck). 
    
**APPROACH:**

    We will consider the prediction of a MW using it's common pitch trajectory. Future versions of this project may consider other elements of the music, such as timbre during a MW segment. 
    
**PROCEEDURE:**

    1) DATA COLLECTION
    2) DATA WRANGLING
    3) DATA EXPLORATION
    4) HYPOTHESIS
    5) DATA MODELLING
    6) MODEL TEST RESULTS AND ACCURACY
    7) DISCUSSION AND LIMITATIONS
    8) CONCLUSION
    9) ARCHIVE
    
**TOOLS:**

    See file: Program.FS

    The majority of data collection and has been done outside the scope of this notebook using web scraping 
    techniques. Data wrangling tools include a functional F# program which uses REST calls to generate tokens 
    and download JSON song information from the Spotify Web API. Various track JSON files were downloaded 
    for exploration including: "Track", "Audio Features" and "Audio Analysis" 

**DATA:**

    See files: nonwhoopsegments.json, whoopsegments.json, allsegments.json (shuffled index)
     
    Original Spotify Audio Analysis JSON files contain much more information that what has been brought 
    into this notebook, and such information is a useful source of embelishment for later versions of 
    this project. 

    The format of the collected audio information that you will see here is of the form 39X216 - where 216 
    is a concatenated array of pitch occurances.  

***

### DATA COLLECTION

Import Python Libraries 


```python
#imports

%matplotlib inline 
%pylab inline 
pylab.rcParams['figure.figsize'] = (12.0, 10.0)

#graphical imports
from IPython.display import Image
import matplotlib as mlp
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import mpl_toolkits.mplot3d as m3d
import seaborn as sns

#numerical, os and dataframe imports
import numpy as np
import os
import pandas as pd

#data modelling imports 
#from sklearn import cross_validation 
import sklearn
from sklearn import datasets
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing 
from sklearn import model_selection
from sklearn import feature_selection
from sklearn.svm import SVC

#stats imports
from scipy import stats
from scipy import stats, special

#file imports
import json
import ijson
from pandas.io.json import json_normalize
```

    Populating the interactive namespace from numpy and matplotlib
    

Explore format of raw Spotify JSON Audio Analysis file, for key 'Segments'


```python
file = 'C:\\Users\\Jnfr\\Projects\\SpotifyAnalyticsApp\\whoopsegments.json'

with open(file) as train_file:
    dict_train = json.load(train_file)

#print(dict_train.head)
df_whoop = pd.DataFrame.from_dict(dict_train)

file = 'C:\\Users\\Jnfr\\Projects\\SpotifyAnalyticsApp\\nonwhoopsegments.json'

with open(file) as train_file:
    dict_train = json.load(train_file)

#print(dict_train.head)
df_nonwhoop = pd.DataFrame.from_dict(dict_train)

file = 'C:\\Users\\Jnfr\\Projects\\SpotifyAnalyticsApp\\allsegments.json'

with open(file) as train_file:
    dict_train = json.load(train_file)

#print(dict_train.head)
df_whoopall = pd.DataFrame.from_dict(dict_train)
```


```python
df_whoop.head()
#del df['id@']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id@</th>
      <th>segmentData@</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6Mk0xtgL6tZ7XQc9GAvobj</td>
      <td>[{'Confidence': 0.128, 'Duration': 0.11156, 'L...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4i1MWIchrtBoSh6wcAdt7e</td>
      <td>[{'Confidence': 0.673, 'Duration': 0.34685, 'L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40ydUI6moZMoOd65wbf6oz</td>
      <td>[{'Confidence': 0.5, 'Duration': 0.2332, 'Loud...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3PYqi5gL3oQMaTPbKZuX7K</td>
      <td>[{'Confidence': 0.311, 'Duration': 0.25324, 'L...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3CrIwWljwzUizLxGpyhclW</td>
      <td>[{'Confidence': 0.545, 'Duration': 0.30912, 'L...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_nonwhoop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id@</th>
      <th>segmentData@</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1yNyoWWWikbLhwIGWjZuDW</td>
      <td>[{'Confidence': 0.864, 'Duration': 0.47116, 'L...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2wqaekenSQZm7hxQOYt8oE</td>
      <td>[{'Confidence': 1, 'Duration': 0.20454, 'Loudn...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5x9VIW2fS21JMswOt6AORI</td>
      <td>[{'Confidence': 0.52, 'Duration': 0.06295, 'Lo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2jyjhRf6DVbMPU5zxagN2h</td>
      <td>[{'Confidence': 0.791, 'Duration': 0.16889, 'L...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2CEgGE6aESpnmtfiZwYlbV</td>
      <td>[{'Confidence': 0.461, 'Duration': 0.33596, 'L...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_whoopall.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id@</th>
      <th>segmentData@</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7C0ISS4UDaBWll6UfRfVrY</td>
      <td>[{'Confidence': 0.516, 'Duration': 0.36558, 'L...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42d6bxB7OQ6f8ct2ihYNBw</td>
      <td>[{'Confidence': 0.303, 'Duration': 0.29596, 'L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64gdv7HDzLmVBUlBf8sGYZ</td>
      <td>[{'Confidence': 0.497, 'Duration': 0.12785, 'L...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3DmW6y7wTEYHJZlLo1r6XJ</td>
      <td>[{'Confidence': 1, 'Duration': 0.23927, 'Loudn...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6g1NlCpW7fgqDnWbCCDrHl</td>
      <td>[{'Confidence': 0.75, 'Duration': 0.6449, 'Lou...</td>
    </tr>
  </tbody>
</table>
</div>



### WRANGLE DATA


```python
#first for whoop songs, df_whoop

#stores 216 song pitches for each segment, concatenated in list of 18 segments/song
###list = []

df_pitches = pd.DataFrame(columns=[x for x in range(216)])

#for index 0...38, row id@, segmentData@
for index, row in df_whoop.iterrows():
    
    for n in range(18):
        if n == 0:
            templist = []
            templist = row['segmentData@'][n]['Pitches']
        elif n <=18:
            templist += row['segmentData@'][n]['Pitches']
    
    ###list+=templist
    
    df_pitches.loc[index] = templist
# list is 12 * 18 * 39 (12 pitches/segment * 18 segments/song * 39 songs)
df_pitches.head()

###print(len(list))
###print(list)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>206</th>
      <th>207</th>
      <th>208</th>
      <th>209</th>
      <th>210</th>
      <th>211</th>
      <th>212</th>
      <th>213</th>
      <th>214</th>
      <th>215</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.564</td>
      <td>0.946</td>
      <td>0.615</td>
      <td>0.502</td>
      <td>0.659</td>
      <td>0.685</td>
      <td>0.814</td>
      <td>0.840</td>
      <td>0.771</td>
      <td>0.783</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.800</td>
      <td>0.286</td>
      <td>0.141</td>
      <td>0.248</td>
      <td>0.139</td>
      <td>0.183</td>
      <td>0.639</td>
      <td>0.235</td>
      <td>0.364</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.183</td>
      <td>1.000</td>
      <td>0.983</td>
      <td>0.377</td>
      <td>0.173</td>
      <td>0.139</td>
      <td>0.276</td>
      <td>0.107</td>
      <td>0.089</td>
      <td>0.084</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.452</td>
      <td>0.227</td>
      <td>0.230</td>
      <td>0.480</td>
      <td>0.241</td>
      <td>0.152</td>
      <td>0.132</td>
      <td>0.256</td>
      <td>0.273</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.489</td>
      <td>0.329</td>
      <td>0.142</td>
      <td>0.774</td>
      <td>0.173</td>
      <td>0.106</td>
      <td>0.228</td>
      <td>0.709</td>
      <td>0.181</td>
      <td>0.165</td>
      <td>...</td>
      <td>0.156</td>
      <td>0.193</td>
      <td>0.386</td>
      <td>0.328</td>
      <td>0.675</td>
      <td>0.674</td>
      <td>0.980</td>
      <td>0.991</td>
      <td>1.000</td>
      <td>0.048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.710</td>
      <td>1.000</td>
      <td>0.387</td>
      <td>0.215</td>
      <td>0.464</td>
      <td>0.350</td>
      <td>0.121</td>
      <td>0.282</td>
      <td>0.142</td>
      <td>0.051</td>
      <td>...</td>
      <td>0.494</td>
      <td>0.292</td>
      <td>0.518</td>
      <td>0.215</td>
      <td>0.407</td>
      <td>1.000</td>
      <td>0.287</td>
      <td>0.093</td>
      <td>0.078</td>
      <td>0.521</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.072</td>
      <td>0.074</td>
      <td>1.000</td>
      <td>0.110</td>
      <td>0.038</td>
      <td>0.218</td>
      <td>0.029</td>
      <td>0.027</td>
      <td>0.028</td>
      <td>0.044</td>
      <td>...</td>
      <td>0.150</td>
      <td>0.109</td>
      <td>0.264</td>
      <td>1.000</td>
      <td>0.237</td>
      <td>0.082</td>
      <td>0.083</td>
      <td>0.108</td>
      <td>0.185</td>
      <td>0.557</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 216 columns</p>
</div>




```python
#second for nonwhoop songs, df_whoop

#stores 216 song pitches for each segment, concatenated in list of 18 segments/song
###list = []

df_pitchesnon = pd.DataFrame(columns=[x for x in range(216)])

#for index 0...38, row id@, segmentData@
for index, row in df_nonwhoop.iterrows():
    
    for n in range(18):
        if n == 0:
            templist = []
            templist = row['segmentData@'][n]['Pitches']
        elif n <=18:
            templist += row['segmentData@'][n]['Pitches']
    
    ###list+=templist
    
    df_pitchesnon.loc[index] = templist
# list is 12 * 18 * 39 (12 pitches/segment * 18 segments/song * 39 songs)
df_pitchesnon.head()

###print(len(list))
###print(list)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>206</th>
      <th>207</th>
      <th>208</th>
      <th>209</th>
      <th>210</th>
      <th>211</th>
      <th>212</th>
      <th>213</th>
      <th>214</th>
      <th>215</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.025</td>
      <td>0.078</td>
      <td>0.617</td>
      <td>1.000</td>
      <td>0.068</td>
      <td>0.037</td>
      <td>0.032</td>
      <td>0.041</td>
      <td>0.017</td>
      <td>0.067</td>
      <td>...</td>
      <td>0.033</td>
      <td>0.107</td>
      <td>0.159</td>
      <td>0.156</td>
      <td>0.918</td>
      <td>0.920</td>
      <td>0.960</td>
      <td>1.000</td>
      <td>0.257</td>
      <td>0.085</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.206</td>
      <td>0.184</td>
      <td>0.487</td>
      <td>1.000</td>
      <td>0.438</td>
      <td>0.254</td>
      <td>0.079</td>
      <td>0.355</td>
      <td>0.671</td>
      <td>0.296</td>
      <td>...</td>
      <td>0.271</td>
      <td>0.258</td>
      <td>0.283</td>
      <td>0.457</td>
      <td>0.312</td>
      <td>0.524</td>
      <td>1.000</td>
      <td>0.654</td>
      <td>0.324</td>
      <td>0.161</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.195</td>
      <td>0.519</td>
      <td>0.404</td>
      <td>0.389</td>
      <td>0.800</td>
      <td>1.000</td>
      <td>0.897</td>
      <td>0.814</td>
      <td>0.459</td>
      <td>0.461</td>
      <td>...</td>
      <td>0.092</td>
      <td>0.135</td>
      <td>0.109</td>
      <td>0.212</td>
      <td>1.000</td>
      <td>0.121</td>
      <td>0.079</td>
      <td>0.058</td>
      <td>0.124</td>
      <td>0.072</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000</td>
      <td>0.256</td>
      <td>0.109</td>
      <td>0.034</td>
      <td>0.013</td>
      <td>0.025</td>
      <td>0.194</td>
      <td>0.076</td>
      <td>0.111</td>
      <td>0.101</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.466</td>
      <td>0.079</td>
      <td>0.009</td>
      <td>0.029</td>
      <td>0.009</td>
      <td>0.008</td>
      <td>0.071</td>
      <td>0.015</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.140</td>
      <td>1.000</td>
      <td>0.333</td>
      <td>0.141</td>
      <td>0.174</td>
      <td>0.084</td>
      <td>0.139</td>
      <td>0.200</td>
      <td>0.394</td>
      <td>0.087</td>
      <td>...</td>
      <td>0.343</td>
      <td>0.243</td>
      <td>0.325</td>
      <td>0.201</td>
      <td>0.509</td>
      <td>0.355</td>
      <td>0.251</td>
      <td>0.218</td>
      <td>0.263</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 216 columns</p>
</div>




```python
#first for whoop songs, df_whoop

#stores 216 song pitches for each segment, concatenated in list of 18 segments/song
###list = []

df_pitchesall = pd.DataFrame(columns=[x for x in range(216)])

#for index 0...38, row id@, segmentData@
for index, row in df_whoopall.iterrows():
    
    for n in range(18):
        if n == 0:
            templist = []
            templist = row['segmentData@'][n]['Pitches']
        elif n <=18:
            templist += row['segmentData@'][n]['Pitches']
    
    ###list+=templist
    
    df_pitchesall.loc[index] = templist
# list is 12 * 18 * 39 (12 pitches/segment * 18 segments/song * 39 songs)
df_pitchesall.head()

###print(len(list))
###print(list)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>206</th>
      <th>207</th>
      <th>208</th>
      <th>209</th>
      <th>210</th>
      <th>211</th>
      <th>212</th>
      <th>213</th>
      <th>214</th>
      <th>215</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.497</td>
      <td>0.187</td>
      <td>0.225</td>
      <td>0.170</td>
      <td>0.202</td>
      <td>1.000</td>
      <td>0.185</td>
      <td>0.296</td>
      <td>0.306</td>
      <td>0.375</td>
      <td>...</td>
      <td>0.159</td>
      <td>0.089</td>
      <td>0.134</td>
      <td>0.408</td>
      <td>0.137</td>
      <td>0.162</td>
      <td>0.176</td>
      <td>0.234</td>
      <td>1.000</td>
      <td>0.339</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.558</td>
      <td>0.314</td>
      <td>1.000</td>
      <td>0.514</td>
      <td>0.293</td>
      <td>0.132</td>
      <td>0.132</td>
      <td>0.166</td>
      <td>0.095</td>
      <td>0.145</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.052</td>
      <td>0.038</td>
      <td>0.029</td>
      <td>0.090</td>
      <td>0.054</td>
      <td>0.362</td>
      <td>0.366</td>
      <td>0.144</td>
      <td>0.088</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.220</td>
      <td>1.000</td>
      <td>0.637</td>
      <td>0.590</td>
      <td>0.319</td>
      <td>0.310</td>
      <td>0.338</td>
      <td>0.167</td>
      <td>0.029</td>
      <td>0.065</td>
      <td>...</td>
      <td>0.854</td>
      <td>0.933</td>
      <td>0.661</td>
      <td>0.654</td>
      <td>0.298</td>
      <td>0.318</td>
      <td>0.190</td>
      <td>0.204</td>
      <td>0.632</td>
      <td>0.161</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.041</td>
      <td>0.466</td>
      <td>1.000</td>
      <td>0.455</td>
      <td>0.106</td>
      <td>0.028</td>
      <td>0.023</td>
      <td>0.018</td>
      <td>0.027</td>
      <td>0.196</td>
      <td>...</td>
      <td>0.230</td>
      <td>0.205</td>
      <td>0.466</td>
      <td>0.467</td>
      <td>0.783</td>
      <td>0.753</td>
      <td>0.986</td>
      <td>1.000</td>
      <td>0.988</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.073</td>
      <td>0.176</td>
      <td>0.268</td>
      <td>0.058</td>
      <td>0.118</td>
      <td>0.153</td>
      <td>0.276</td>
      <td>0.236</td>
      <td>0.323</td>
      <td>1.000</td>
      <td>...</td>
      <td>0.430</td>
      <td>0.161</td>
      <td>0.158</td>
      <td>0.173</td>
      <td>0.439</td>
      <td>0.253</td>
      <td>0.158</td>
      <td>0.445</td>
      <td>0.268</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 216 columns</p>
</div>



### DATA EXPLORATION

#### 1. Visualizing instances where we know the pitch segments contain "whoops" 

Visualizing whoop pitch patterns over a span of 18 segments between two songs:

*Though the timing information has been stripped for sake of simplicity this span of segments is equivalent to ~3-4 seconds in duration, depending on the song.*


```python
df_pitches.loc[0].plot()
df_pitches.loc[1].plot()
df_pitches.loc[2].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2230f0d0da0>




![png](output_20_1.png)



```python
df_pitchesnon.loc[0].plot()
df_pitchesnon.loc[1].plot()
df_pitchesnon.loc[2].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2230f469518>




![png](output_21_1.png)


Vizualizing overlay of pitch range across 39 "whoop" songs. Pitch frequencies at the lower end and higher end (0...0.3 and 1.0) are most popular accross all songs as expected. This is because 1.0 represents a pure tone (note) in a given song segment, meanwhile smaller values represent a small amount of a given note, likely occuring due to fade out from a pure tone in a previous segment.


```python
df_pitches.plot(kind='hist', legend=None)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2230f4b6da0>




![png](output_23_1.png)



```python
df_pitchesnon.plot(kind='hist', legend=None)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2230f5c7be0>




![png](output_24_1.png)


Question: How do the pitches themselves accross song segments correlate to one another?


```python
print(df_pitches.corr())
```

              0         1         2         3         4         5         6    \
    0    1.000000  0.490782 -0.043912  0.268518  0.381535  0.171231  0.029962   
    1    0.490782  1.000000  0.279150  0.361944  0.381968  0.068460  0.061122   
    2   -0.043912  0.279150  1.000000  0.424215  0.199305  0.003563  0.043363   
    3    0.268518  0.361944  0.424215  1.000000  0.678191  0.346510  0.218663   
    4    0.381535  0.381968  0.199305  0.678191  1.000000  0.716095  0.387946   
    5    0.171231  0.068460  0.003563  0.346510  0.716095  1.000000  0.374331   
    6    0.029962  0.061122  0.043363  0.218663  0.387946  0.374331  1.000000   
    7   -0.023547 -0.147446  0.030992  0.088515  0.168363  0.080578  0.512474   
    8    0.292300  0.371711 -0.039463  0.312026  0.403915  0.243346  0.451186   
    9    0.121582  0.149205 -0.085339  0.171047  0.362992  0.232323  0.384508   
    10   0.125864  0.093250 -0.345322  0.136528  0.062700  0.028788  0.058975   
    11   0.384166  0.187015 -0.001446  0.305965  0.275725  0.133809 -0.092220   
    12   0.296013  0.281937  0.330100  0.236293  0.198281  0.088615  0.246780   
    13   0.243212  0.559455  0.346081  0.343256  0.232577  0.023585  0.192656   
    14   0.186676  0.223010  0.088643  0.195682  0.049656  0.031048 -0.060856   
    15   0.327285  0.312917 -0.088318  0.368850  0.120670  0.093012 -0.038506   
    16   0.222950  0.430796  0.180420  0.320810  0.291977  0.148672 -0.042156   
    17   0.019075  0.037367 -0.085039  0.005166  0.124426  0.360167 -0.005687   
    18  -0.061264  0.044874 -0.154363  0.037088  0.070924  0.079210  0.110515   
    19  -0.072500 -0.033538 -0.090616  0.030323 -0.024542 -0.090243  0.097671   
    20   0.196607  0.190127 -0.112619  0.033182  0.071341  0.114198  0.249657   
    21  -0.044546  0.191387  0.212089  0.134122  0.001283  0.011111 -0.001980   
    22  -0.008225  0.060322 -0.039726  0.072086 -0.143688 -0.044995 -0.116383   
    23   0.325647  0.207792  0.100424  0.065309 -0.052246 -0.005435 -0.159906   
    24   0.254953  0.511840  0.099312  0.254221  0.263911  0.087385  0.051466   
    25   0.522269  0.721178  0.201366  0.400836  0.308453  0.011514  0.212310   
    26   0.254856  0.371589  0.085912  0.296069  0.053073 -0.238380  0.189623   
    27   0.248552  0.344347  0.039832  0.259158  0.155513 -0.074284  0.378891   
    28   0.195414  0.406120  0.284038  0.259867  0.272291  0.142177  0.187328   
    29  -0.019024  0.128435  0.132822  0.145118 -0.051449  0.054826 -0.105020   
    ..        ...       ...       ...       ...       ...       ...       ...   
    186 -0.129458 -0.066846 -0.127945 -0.068985 -0.244451 -0.213501 -0.068722   
    187 -0.128765 -0.062237 -0.040904 -0.065196 -0.171421 -0.068143  0.007803   
    188  0.002107  0.184009 -0.033374  0.031781 -0.114764 -0.132194 -0.070463   
    189  0.085183  0.246830  0.185501  0.100848  0.086113 -0.017141  0.070753   
    190  0.138815  0.219453  0.071377 -0.027898 -0.099019 -0.079312 -0.091234   
    191  0.237486  0.186531 -0.086092 -0.024245  0.138197  0.067670 -0.025135   
    192  0.265687  0.105897 -0.115728  0.102180  0.291746  0.242471  0.209002   
    193  0.122196  0.486188  0.155972  0.299311  0.411715  0.188010  0.337740   
    194  0.108440  0.147695  0.201549  0.270082  0.170115  0.113739  0.183993   
    195  0.114040  0.173199 -0.149876  0.249610  0.134031  0.110250  0.208864   
    196  0.096951  0.110509 -0.094934  0.127824  0.137016  0.073640  0.123321   
    197 -0.081680 -0.188821 -0.083991 -0.208533 -0.066309  0.198191 -0.365324   
    198 -0.026045  0.301646 -0.137978  0.058420 -0.061696 -0.130871 -0.259159   
    199  0.112352  0.176119 -0.228966 -0.294431 -0.204353 -0.172135 -0.152261   
    200 -0.001816  0.161810 -0.193695 -0.180008 -0.221049 -0.237885  0.028411   
    201  0.034966  0.203529  0.203235  0.176186  0.062429  0.069846  0.061137   
    202  0.290097  0.409497 -0.052498  0.124063  0.018439  0.018056 -0.017668   
    203  0.375334  0.328467 -0.169801  0.063522  0.152565  0.154003 -0.044096   
    204  0.332756  0.182907 -0.012459  0.062043 -0.082797 -0.135571 -0.130162   
    205  0.237855  0.630483  0.114277  0.424800  0.126117 -0.138953  0.092379   
    206 -0.083281  0.327719  0.304442  0.177984  0.015772 -0.212061 -0.001417   
    207  0.048894  0.440534  0.042269  0.219903  0.126304 -0.069757  0.169133   
    208  0.023714  0.397474 -0.073475  0.056157  0.139454 -0.012554  0.186944   
    209 -0.094311  0.048823  0.061035 -0.065033  0.041445  0.290374 -0.175736   
    210  0.185558  0.544875  0.011581  0.306294  0.138993 -0.040045 -0.004738   
    211  0.098986  0.352386 -0.094162  0.121033  0.106106 -0.084530  0.049073   
    212  0.115661  0.341393 -0.160342  0.205476 -0.011653 -0.168095  0.254661   
    213  0.265155  0.261229 -0.027095  0.295890  0.166942 -0.067812  0.247849   
    214  0.218356  0.103783 -0.076491  0.212221 -0.093253 -0.022790 -0.135748   
    215  0.262060  0.302413  0.127707 -0.090814 -0.200509 -0.125898 -0.149585   
    
              7         8         9      ...          206       207       208  \
    0   -0.023547  0.292300  0.121582    ...    -0.083281  0.048894  0.023714   
    1   -0.147446  0.371711  0.149205    ...     0.327719  0.440534  0.397474   
    2    0.030992 -0.039463 -0.085339    ...     0.304442  0.042269 -0.073475   
    3    0.088515  0.312026  0.171047    ...     0.177984  0.219903  0.056157   
    4    0.168363  0.403915  0.362992    ...     0.015772  0.126304  0.139454   
    5    0.080578  0.243346  0.232323    ...    -0.212061 -0.069757 -0.012554   
    6    0.512474  0.451186  0.384508    ...    -0.001417  0.169133  0.186944   
    7    1.000000  0.334503  0.189442    ...    -0.130839 -0.008330  0.082481   
    8    0.334503  1.000000  0.691792    ...     0.018221  0.233767  0.393816   
    9    0.189442  0.691792  1.000000    ...    -0.004515  0.154340  0.439106   
    10   0.005218  0.350858  0.399031    ...     0.232356  0.416865  0.265015   
    11  -0.200830 -0.012463 -0.041680    ...     0.245995  0.021013 -0.131533   
    12  -0.093631 -0.056558 -0.081572    ...    -0.026385  0.121118  0.025080   
    13  -0.120170  0.281608  0.169563    ...     0.252385  0.295904  0.159766   
    14   0.031363  0.219743  0.141062    ...     0.227451  0.067579 -0.083237   
    15   0.012868  0.193420  0.109453    ...     0.130707  0.165615  0.122454   
    16  -0.040309  0.262959  0.129816    ...     0.342802  0.215831  0.249240   
    17  -0.229485  0.078788 -0.018375    ...    -0.180891 -0.090745 -0.004860   
    18  -0.075406  0.268138  0.306790    ...    -0.053389  0.050045  0.302124   
    19   0.529822  0.302292  0.072311    ...    -0.085406 -0.008697  0.109082   
    20   0.324401  0.565159  0.349862    ...    -0.165889 -0.029363  0.283031   
    21   0.124089  0.168903  0.172045    ...     0.173834  0.117732  0.276453   
    22  -0.024993 -0.014970  0.046350    ...     0.088235  0.303132  0.211102   
    23  -0.329088 -0.162044  0.030117    ...     0.129237 -0.054812 -0.217512   
    24  -0.006459  0.198990 -0.057360    ...     0.255733  0.377773  0.273635   
    25   0.051319  0.482619  0.212669    ...     0.249156  0.373246  0.333367   
    26  -0.046193  0.270159  0.147937    ...     0.084615  0.159910  0.026190   
    27   0.074440  0.300390  0.086506    ...     0.112277  0.134135  0.052067   
    28   0.054136  0.173727  0.013107    ...     0.175347  0.122282  0.229046   
    29  -0.205675  0.054213 -0.087410    ...    -0.078537  0.097028  0.058216   
    ..        ...       ...       ...    ...          ...       ...       ...   
    186 -0.343078 -0.003922  0.052885    ...    -0.052223 -0.274642 -0.179421   
    187  0.043162  0.052495  0.128325    ...    -0.185494 -0.141876 -0.145401   
    188 -0.028630 -0.071970 -0.002727    ...     0.096979  0.318235  0.133838   
    189  0.065442  0.055389  0.198542    ...     0.109910  0.202114  0.096906   
    190 -0.006359 -0.118264 -0.053580    ...     0.050173  0.191614  0.055951   
    191 -0.007460 -0.075393 -0.047058    ...    -0.116160 -0.015947  0.138408   
    192  0.309652  0.011825 -0.024127    ...    -0.006058  0.380664  0.315235   
    193  0.216504  0.300255  0.277597    ...     0.438606  0.543586  0.519343   
    194  0.103007  0.209790  0.276954    ...     0.479299  0.280659  0.184127   
    195  0.136990  0.225989  0.223278    ...     0.254748  0.384520  0.394963   
    196  0.182001  0.142282  0.206427    ...     0.185750  0.194676  0.428883   
    197 -0.180639 -0.238810 -0.152762    ...    -0.254827 -0.294160 -0.089855   
    198 -0.214649  0.213078  0.235392    ...     0.245734  0.015576  0.253235   
    199  0.073939  0.101461  0.173869    ...    -0.015961  0.062014  0.140530   
    200 -0.224013 -0.082442 -0.088745    ...     0.158435  0.522993  0.284332   
    201 -0.113871  0.135963  0.231582    ...    -0.116013  0.134047  0.332839   
    202 -0.177768  0.174002  0.085936    ...     0.049469  0.429476  0.478375   
    203  0.019879  0.074288  0.074299    ...     0.119578  0.258935  0.201265   
    204 -0.031443 -0.189232 -0.269393    ...    -0.038107  0.280561 -0.039216   
    205 -0.040102  0.377993  0.158498    ...     0.291492  0.512418  0.249843   
    206 -0.130839  0.018221 -0.004515    ...     1.000000  0.546359  0.128699   
    207 -0.008330  0.233767  0.154340    ...     0.546359  1.000000  0.618084   
    208  0.082481  0.393816  0.439106    ...     0.128699  0.618084  1.000000   
    209 -0.096486 -0.085234 -0.048135    ...    -0.323197 -0.086322  0.176872   
    210  0.024851  0.373496  0.119389    ...     0.157466  0.282796  0.456227   
    211  0.169046  0.226604  0.075571    ...     0.135341  0.120212  0.352124   
    212 -0.006839  0.172349  0.001872    ...     0.114980  0.174392  0.276879   
    213  0.205331  0.509021  0.449090    ...     0.157731  0.359421  0.424727   
    214 -0.086468  0.066185  0.038103    ...    -0.033471  0.155612  0.071115   
    215 -0.213743 -0.114456 -0.081951    ...     0.184275  0.398332  0.108705   
    
              209       210       211       212       213       214       215  
    0   -0.094311  0.185558  0.098986  0.115661  0.265155  0.218356  0.262060  
    1    0.048823  0.544875  0.352386  0.341393  0.261229  0.103783  0.302413  
    2    0.061035  0.011581 -0.094162 -0.160342 -0.027095 -0.076491  0.127707  
    3   -0.065033  0.306294  0.121033  0.205476  0.295890  0.212221 -0.090814  
    4    0.041445  0.138993  0.106106 -0.011653  0.166942 -0.093253 -0.200509  
    5    0.290374 -0.040045 -0.084530 -0.168095 -0.067812 -0.022790 -0.125898  
    6   -0.175736 -0.004738  0.049073  0.254661  0.247849 -0.135748 -0.149585  
    7   -0.096486  0.024851  0.169046 -0.006839  0.205331 -0.086468 -0.213743  
    8   -0.085234  0.373496  0.226604  0.172349  0.509021  0.066185 -0.114456  
    9   -0.048135  0.119389  0.075571  0.001872  0.449090  0.038103 -0.081951  
    10  -0.139753  0.127538  0.101477  0.182136  0.525410  0.539271  0.081833  
    11  -0.187202 -0.151973 -0.022093 -0.136799 -0.145728  0.018014  0.208058  
    12   0.004127  0.072327  0.005019  0.349023  0.348721  0.308005  0.159091  
    13  -0.091128  0.224696  0.075160  0.408785  0.471850  0.301688  0.133517  
    14  -0.001201  0.231607  0.142571  0.128106  0.320100  0.419790  0.161783  
    15  -0.009758  0.379450  0.293816  0.401641  0.441715  0.698517  0.110333  
    16   0.178702  0.416890  0.412495  0.232148  0.265001  0.164737  0.056182  
    17   0.485199  0.151290  0.195044  0.091677 -0.035974  0.053111  0.131742  
    18   0.190641  0.357132  0.148983  0.170372  0.032247  0.048204 -0.056245  
    19   0.196233  0.369898  0.265198  0.136443  0.242634  0.093131 -0.161742  
    20   0.156255  0.348752  0.220450  0.257069  0.417882  0.284482 -0.026469  
    21   0.143995  0.202813  0.298858  0.150490  0.144751  0.192108  0.191051  
    22   0.047358  0.099218  0.122723  0.184058  0.365847  0.664152  0.218551  
    23  -0.012532 -0.073524 -0.043934 -0.058555  0.033835  0.140044  0.300472  
    24   0.053720  0.585674  0.375889  0.394291  0.250887  0.182090  0.102118  
    25  -0.093132  0.621457  0.347372  0.495069  0.559642  0.314219  0.129924  
    26  -0.192947  0.294058  0.215440  0.490926  0.522673  0.325799  0.034390  
    27  -0.150560  0.373151  0.385694  0.643457  0.487765  0.275734  0.007253  
    28   0.141410  0.298470  0.373458  0.277868  0.266304 -0.043623  0.104396  
    29   0.373502  0.346567  0.152912  0.146633  0.067221  0.229987  0.227172  
    ..        ...       ...       ...       ...       ...       ...       ...  
    186 -0.019216  0.147124  0.033014  0.328697 -0.044057 -0.028161 -0.059594  
    187  0.103406 -0.068307  0.253302  0.115270  0.188162 -0.023395  0.033868  
    188  0.272607  0.188585  0.180630  0.248109  0.252349  0.213768  0.288316  
    189  0.259922  0.006613 -0.006908  0.179977  0.346020  0.130690  0.192872  
    190  0.303929 -0.031077 -0.020728  0.108689  0.150529  0.188414  0.505678  
    191  0.158860  0.069400  0.184993 -0.048085  0.003822 -0.157861 -0.083180  
    192 -0.030002 -0.064298 -0.044334 -0.010762  0.146171  0.163550  0.064977  
    193 -0.144001  0.267340  0.261959  0.157232  0.273692  0.116343 -0.009023  
    194 -0.253909  0.074929  0.120277  0.140653  0.243937  0.211285 -0.024131  
    195 -0.107300  0.218805  0.281872  0.345499  0.421298  0.446472 -0.012842  
    196 -0.042010  0.120764  0.272281  0.213416  0.291988  0.237525 -0.048637  
    197  0.659609  0.013632  0.004008 -0.194535 -0.178356 -0.090413  0.023476  
    198  0.044726  0.591181  0.389634  0.263126  0.056663  0.103766  0.015113  
    199 -0.160605  0.018444  0.287457  0.114775  0.173744  0.124367  0.231024  
    200 -0.169313  0.054437  0.023383  0.310513  0.194492  0.196449  0.336079  
    201  0.230312 -0.036816 -0.088559  0.083855  0.304198  0.088912  0.142424  
    202  0.086425  0.257753  0.094272  0.367939  0.435775  0.454385  0.277372  
    203  0.084934  0.268950  0.224809  0.161577  0.142291  0.422658  0.207974  
    204  0.033129  0.035779 -0.226001 -0.034377 -0.073655  0.243257  0.527404  
    205 -0.116933  0.457623  0.090430  0.318156  0.270914  0.239094  0.286742  
    206 -0.323197  0.157466  0.135341  0.114980  0.157731 -0.033471  0.184275  
    207 -0.086322  0.282796  0.120212  0.174392  0.359421  0.155612  0.398332  
    208  0.176872  0.456227  0.352124  0.276879  0.424727  0.071115  0.108705  
    209  1.000000  0.321329  0.067612 -0.033806 -0.029400 -0.099915  0.066029  
    210  0.321329  1.000000  0.672762  0.554850  0.361036  0.242764 -0.055935  
    211  0.067612  0.672762  1.000000  0.508965  0.341981  0.098325 -0.121714  
    212 -0.033806  0.554850  0.508965  1.000000  0.572097  0.405705 -0.091850  
    213 -0.029400  0.361036  0.341981  0.572097  1.000000  0.490779 -0.097614  
    214 -0.099915  0.242764  0.098325  0.405705  0.490779  1.000000  0.064906  
    215  0.066029 -0.055935 -0.121714 -0.091850 -0.097614  0.064906  1.000000  
    
    [216 rows x 216 columns]
    


```python
# visualize coorelation for segments containing a whoop
# we would expect these to coorelate
sns.heatmap(df_pitches.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x223120c2f60>




![png](output_27_1.png)



```python
# visualize coorelation for segments containing that do not, 
#          contain a whoop and have been chosen at random
# we would expect these *not* coorelate
sns.heatmap(df_pitchesnon.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22312f2fd30>




![png](output_28_1.png)



```python
# visualize coorelation for all random and 'whoop' containing segments
# we would expect whoop segments in data to coorelate, 
#        but also to see much noise from the randomly selected segments
sns.heatmap(df_pitchesall.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2231312aa90>




![png](output_29_1.png)


#### 2. Visualizing "non-whoop" segments

### HYPOTHESIS

There is enough evidence to suggest that we may be able to predict the occurance of a "whoop" using pitches over time. 
* Note we could also choose to explore here timbre patterns similarily to how we've explored pitch patterns for whoops and non-whoops. Comparing a model using pitch segments vs timbre segments vs a combination of pitch and timbre information would be useful if time permitted.

### MODEL DATA: SVC

Note: This section has not been completed due to time constraints. allsegments.json would be uploaded into a dataframe and used to fix the below issue of having a monomial classifier (in allsegments.json we have 0 and 1 whoop and non-whoop pitch segments)


```python
#add boolean whoop column to each whoop/non-whoop df and concatenate frames
df_pitches['whoop']= '1'
df_pitchesnon['whoop'] = '0'
print("# samples whoop = 1" + str(len(df_pitches)))
print("# samples whoop = 0" + str(len(df_pitchesnon)))
print("prediction baseline 100/139 = " + str(100/139))
df_all = pd.concat([df_pitches, df_pitchesnon])
```

    # samples whoop = 139
    # samples whoop = 0100
    prediction baseline 100/139 = 0.7194244604316546
    


```python
#shuffle index at random for model
#from sklearn.utils import shuffle
#df_all = shuffle(df_all)
df_all
df_alltrain = df_all[10:70]
df_alltest = df_all[0:10].append(df_all[70:])
df_alltest
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>207</th>
      <th>208</th>
      <th>209</th>
      <th>210</th>
      <th>211</th>
      <th>212</th>
      <th>213</th>
      <th>214</th>
      <th>215</th>
      <th>whoop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.564</td>
      <td>0.946</td>
      <td>0.615</td>
      <td>0.502</td>
      <td>0.659</td>
      <td>0.685</td>
      <td>0.814</td>
      <td>0.840</td>
      <td>0.771</td>
      <td>0.783</td>
      <td>...</td>
      <td>0.800</td>
      <td>0.286</td>
      <td>0.141</td>
      <td>0.248</td>
      <td>0.139</td>
      <td>0.183</td>
      <td>0.639</td>
      <td>0.235</td>
      <td>0.364</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.183</td>
      <td>1.000</td>
      <td>0.983</td>
      <td>0.377</td>
      <td>0.173</td>
      <td>0.139</td>
      <td>0.276</td>
      <td>0.107</td>
      <td>0.089</td>
      <td>0.084</td>
      <td>...</td>
      <td>0.452</td>
      <td>0.227</td>
      <td>0.230</td>
      <td>0.480</td>
      <td>0.241</td>
      <td>0.152</td>
      <td>0.132</td>
      <td>0.256</td>
      <td>0.273</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.489</td>
      <td>0.329</td>
      <td>0.142</td>
      <td>0.774</td>
      <td>0.173</td>
      <td>0.106</td>
      <td>0.228</td>
      <td>0.709</td>
      <td>0.181</td>
      <td>0.165</td>
      <td>...</td>
      <td>0.193</td>
      <td>0.386</td>
      <td>0.328</td>
      <td>0.675</td>
      <td>0.674</td>
      <td>0.980</td>
      <td>0.991</td>
      <td>1.000</td>
      <td>0.048</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.710</td>
      <td>1.000</td>
      <td>0.387</td>
      <td>0.215</td>
      <td>0.464</td>
      <td>0.350</td>
      <td>0.121</td>
      <td>0.282</td>
      <td>0.142</td>
      <td>0.051</td>
      <td>...</td>
      <td>0.292</td>
      <td>0.518</td>
      <td>0.215</td>
      <td>0.407</td>
      <td>1.000</td>
      <td>0.287</td>
      <td>0.093</td>
      <td>0.078</td>
      <td>0.521</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.072</td>
      <td>0.074</td>
      <td>1.000</td>
      <td>0.110</td>
      <td>0.038</td>
      <td>0.218</td>
      <td>0.029</td>
      <td>0.027</td>
      <td>0.028</td>
      <td>0.044</td>
      <td>...</td>
      <td>0.109</td>
      <td>0.264</td>
      <td>1.000</td>
      <td>0.237</td>
      <td>0.082</td>
      <td>0.083</td>
      <td>0.108</td>
      <td>0.185</td>
      <td>0.557</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.231</td>
      <td>0.275</td>
      <td>0.639</td>
      <td>0.127</td>
      <td>0.065</td>
      <td>0.079</td>
      <td>0.360</td>
      <td>1.000</td>
      <td>0.103</td>
      <td>0.227</td>
      <td>...</td>
      <td>0.061</td>
      <td>0.225</td>
      <td>0.071</td>
      <td>0.182</td>
      <td>1.000</td>
      <td>0.119</td>
      <td>0.023</td>
      <td>0.052</td>
      <td>0.229</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.397</td>
      <td>0.642</td>
      <td>1.000</td>
      <td>0.749</td>
      <td>0.580</td>
      <td>0.871</td>
      <td>0.358</td>
      <td>0.214</td>
      <td>0.147</td>
      <td>0.209</td>
      <td>...</td>
      <td>0.057</td>
      <td>0.265</td>
      <td>1.000</td>
      <td>0.304</td>
      <td>0.060</td>
      <td>0.038</td>
      <td>0.114</td>
      <td>0.053</td>
      <td>0.086</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.753</td>
      <td>0.771</td>
      <td>0.675</td>
      <td>0.588</td>
      <td>0.971</td>
      <td>0.909</td>
      <td>0.886</td>
      <td>1.000</td>
      <td>0.428</td>
      <td>0.424</td>
      <td>...</td>
      <td>0.276</td>
      <td>0.687</td>
      <td>0.599</td>
      <td>0.857</td>
      <td>1.000</td>
      <td>0.504</td>
      <td>0.498</td>
      <td>0.076</td>
      <td>0.034</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.117</td>
      <td>0.160</td>
      <td>0.084</td>
      <td>0.087</td>
      <td>0.191</td>
      <td>0.096</td>
      <td>0.142</td>
      <td>0.125</td>
      <td>0.224</td>
      <td>1.000</td>
      <td>...</td>
      <td>0.228</td>
      <td>1.000</td>
      <td>0.259</td>
      <td>0.240</td>
      <td>0.069</td>
      <td>0.048</td>
      <td>0.065</td>
      <td>0.057</td>
      <td>0.152</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.120</td>
      <td>0.021</td>
      <td>0.118</td>
      <td>0.018</td>
      <td>0.041</td>
      <td>0.079</td>
      <td>0.101</td>
      <td>0.021</td>
      <td>0.040</td>
      <td>0.247</td>
      <td>...</td>
      <td>0.099</td>
      <td>0.051</td>
      <td>0.088</td>
      <td>0.124</td>
      <td>0.340</td>
      <td>0.072</td>
      <td>0.126</td>
      <td>0.026</td>
      <td>0.195</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.998</td>
      <td>0.209</td>
      <td>0.160</td>
      <td>0.483</td>
      <td>0.135</td>
      <td>1.000</td>
      <td>0.212</td>
      <td>0.246</td>
      <td>0.310</td>
      <td>0.236</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.086</td>
      <td>0.080</td>
      <td>0.055</td>
      <td>0.229</td>
      <td>0.042</td>
      <td>0.068</td>
      <td>0.296</td>
      <td>0.126</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.439</td>
      <td>0.067</td>
      <td>0.081</td>
      <td>0.095</td>
      <td>0.107</td>
      <td>0.076</td>
      <td>0.083</td>
      <td>0.169</td>
      <td>0.093</td>
      <td>0.094</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.552</td>
      <td>0.055</td>
      <td>0.112</td>
      <td>0.131</td>
      <td>0.040</td>
      <td>0.040</td>
      <td>0.062</td>
      <td>0.082</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.376</td>
      <td>1.000</td>
      <td>0.635</td>
      <td>0.316</td>
      <td>0.267</td>
      <td>0.264</td>
      <td>0.295</td>
      <td>0.179</td>
      <td>0.222</td>
      <td>0.415</td>
      <td>...</td>
      <td>0.079</td>
      <td>0.245</td>
      <td>0.250</td>
      <td>0.381</td>
      <td>0.128</td>
      <td>0.350</td>
      <td>1.000</td>
      <td>0.242</td>
      <td>0.125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.050</td>
      <td>0.086</td>
      <td>1.000</td>
      <td>0.209</td>
      <td>0.429</td>
      <td>0.184</td>
      <td>0.416</td>
      <td>0.100</td>
      <td>0.042</td>
      <td>0.167</td>
      <td>...</td>
      <td>0.151</td>
      <td>0.070</td>
      <td>0.032</td>
      <td>0.128</td>
      <td>0.123</td>
      <td>0.053</td>
      <td>0.097</td>
      <td>0.035</td>
      <td>0.116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.000</td>
      <td>0.273</td>
      <td>0.190</td>
      <td>0.219</td>
      <td>0.369</td>
      <td>0.977</td>
      <td>0.164</td>
      <td>0.219</td>
      <td>0.361</td>
      <td>0.142</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.392</td>
      <td>0.386</td>
      <td>0.181</td>
      <td>0.208</td>
      <td>0.399</td>
      <td>0.418</td>
      <td>0.837</td>
      <td>0.295</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.543</td>
      <td>1.000</td>
      <td>0.092</td>
      <td>0.076</td>
      <td>0.131</td>
      <td>0.118</td>
      <td>0.226</td>
      <td>0.065</td>
      <td>0.042</td>
      <td>0.195</td>
      <td>...</td>
      <td>0.038</td>
      <td>0.067</td>
      <td>0.113</td>
      <td>0.332</td>
      <td>0.051</td>
      <td>0.071</td>
      <td>0.047</td>
      <td>0.062</td>
      <td>0.059</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.593</td>
      <td>1.000</td>
      <td>0.821</td>
      <td>0.668</td>
      <td>0.700</td>
      <td>0.498</td>
      <td>0.708</td>
      <td>0.396</td>
      <td>0.332</td>
      <td>0.530</td>
      <td>...</td>
      <td>0.183</td>
      <td>0.249</td>
      <td>0.159</td>
      <td>0.152</td>
      <td>0.133</td>
      <td>0.251</td>
      <td>0.264</td>
      <td>0.172</td>
      <td>0.047</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1.000</td>
      <td>0.755</td>
      <td>0.245</td>
      <td>0.139</td>
      <td>0.174</td>
      <td>0.165</td>
      <td>0.235</td>
      <td>0.403</td>
      <td>0.491</td>
      <td>0.430</td>
      <td>...</td>
      <td>0.354</td>
      <td>0.403</td>
      <td>0.268</td>
      <td>0.668</td>
      <td>1.000</td>
      <td>0.423</td>
      <td>0.498</td>
      <td>0.293</td>
      <td>0.940</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.548</td>
      <td>1.000</td>
      <td>0.128</td>
      <td>0.145</td>
      <td>0.247</td>
      <td>0.069</td>
      <td>0.112</td>
      <td>0.112</td>
      <td>0.651</td>
      <td>0.037</td>
      <td>...</td>
      <td>0.277</td>
      <td>1.000</td>
      <td>0.270</td>
      <td>0.160</td>
      <td>0.110</td>
      <td>0.179</td>
      <td>0.082</td>
      <td>0.119</td>
      <td>0.455</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.071</td>
      <td>0.029</td>
      <td>0.017</td>
      <td>0.059</td>
      <td>1.000</td>
      <td>0.100</td>
      <td>0.093</td>
      <td>0.251</td>
      <td>0.195</td>
      <td>0.507</td>
      <td>...</td>
      <td>0.007</td>
      <td>0.268</td>
      <td>0.006</td>
      <td>0.006</td>
      <td>0.006</td>
      <td>0.072</td>
      <td>1.000</td>
      <td>0.098</td>
      <td>0.006</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.120</td>
      <td>0.374</td>
      <td>0.335</td>
      <td>0.356</td>
      <td>0.874</td>
      <td>0.809</td>
      <td>1.000</td>
      <td>0.866</td>
      <td>0.415</td>
      <td>0.425</td>
      <td>...</td>
      <td>0.231</td>
      <td>0.766</td>
      <td>0.476</td>
      <td>0.955</td>
      <td>1.000</td>
      <td>0.527</td>
      <td>0.613</td>
      <td>0.363</td>
      <td>0.318</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.916</td>
      <td>0.343</td>
      <td>0.062</td>
      <td>0.035</td>
      <td>0.046</td>
      <td>0.039</td>
      <td>0.102</td>
      <td>0.152</td>
      <td>0.479</td>
      <td>0.482</td>
      <td>...</td>
      <td>0.145</td>
      <td>0.167</td>
      <td>0.298</td>
      <td>0.486</td>
      <td>0.542</td>
      <td>0.218</td>
      <td>0.156</td>
      <td>0.138</td>
      <td>1.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1.000</td>
      <td>0.384</td>
      <td>0.083</td>
      <td>0.165</td>
      <td>0.130</td>
      <td>0.447</td>
      <td>0.164</td>
      <td>0.127</td>
      <td>0.098</td>
      <td>0.222</td>
      <td>...</td>
      <td>0.071</td>
      <td>0.050</td>
      <td>0.058</td>
      <td>0.351</td>
      <td>0.896</td>
      <td>0.249</td>
      <td>0.156</td>
      <td>0.227</td>
      <td>1.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.192</td>
      <td>0.527</td>
      <td>0.068</td>
      <td>0.107</td>
      <td>0.266</td>
      <td>0.068</td>
      <td>0.161</td>
      <td>0.077</td>
      <td>0.106</td>
      <td>0.481</td>
      <td>...</td>
      <td>0.387</td>
      <td>0.336</td>
      <td>0.467</td>
      <td>0.458</td>
      <td>0.651</td>
      <td>0.548</td>
      <td>0.827</td>
      <td>0.593</td>
      <td>0.285</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.041</td>
      <td>0.466</td>
      <td>1.000</td>
      <td>0.455</td>
      <td>0.106</td>
      <td>0.028</td>
      <td>0.023</td>
      <td>0.018</td>
      <td>0.027</td>
      <td>0.196</td>
      <td>...</td>
      <td>0.205</td>
      <td>0.466</td>
      <td>0.467</td>
      <td>0.783</td>
      <td>0.753</td>
      <td>0.986</td>
      <td>1.000</td>
      <td>0.988</td>
      <td>0.026</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.158</td>
      <td>0.119</td>
      <td>0.082</td>
      <td>0.140</td>
      <td>0.132</td>
      <td>0.101</td>
      <td>0.068</td>
      <td>1.000</td>
      <td>0.220</td>
      <td>0.045</td>
      <td>...</td>
      <td>0.093</td>
      <td>0.189</td>
      <td>1.000</td>
      <td>0.719</td>
      <td>0.113</td>
      <td>0.066</td>
      <td>0.157</td>
      <td>0.033</td>
      <td>0.041</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1.000</td>
      <td>0.754</td>
      <td>0.067</td>
      <td>0.063</td>
      <td>0.142</td>
      <td>0.159</td>
      <td>0.330</td>
      <td>0.333</td>
      <td>0.748</td>
      <td>0.730</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.489</td>
      <td>0.181</td>
      <td>0.203</td>
      <td>0.396</td>
      <td>0.800</td>
      <td>0.137</td>
      <td>0.149</td>
      <td>0.088</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.226</td>
      <td>0.160</td>
      <td>0.161</td>
      <td>0.243</td>
      <td>0.460</td>
      <td>0.337</td>
      <td>0.248</td>
      <td>1.000</td>
      <td>0.353</td>
      <td>0.112</td>
      <td>...</td>
      <td>0.522</td>
      <td>0.359</td>
      <td>0.234</td>
      <td>0.478</td>
      <td>0.631</td>
      <td>0.281</td>
      <td>0.245</td>
      <td>0.189</td>
      <td>1.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.852</td>
      <td>0.357</td>
      <td>0.425</td>
      <td>0.286</td>
      <td>0.366</td>
      <td>0.338</td>
      <td>0.448</td>
      <td>0.694</td>
      <td>0.401</td>
      <td>0.707</td>
      <td>...</td>
      <td>0.215</td>
      <td>0.263</td>
      <td>0.137</td>
      <td>0.163</td>
      <td>0.188</td>
      <td>0.072</td>
      <td>0.414</td>
      <td>0.106</td>
      <td>0.239</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.358</td>
      <td>0.399</td>
      <td>0.531</td>
      <td>0.804</td>
      <td>0.247</td>
      <td>0.417</td>
      <td>1.000</td>
      <td>0.250</td>
      <td>0.302</td>
      <td>0.350</td>
      <td>...</td>
      <td>0.383</td>
      <td>0.311</td>
      <td>0.198</td>
      <td>0.314</td>
      <td>0.557</td>
      <td>1.000</td>
      <td>0.302</td>
      <td>0.177</td>
      <td>0.134</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.038</td>
      <td>0.361</td>
      <td>0.038</td>
      <td>0.027</td>
      <td>0.182</td>
      <td>0.197</td>
      <td>1.000</td>
      <td>0.142</td>
      <td>0.047</td>
      <td>0.109</td>
      <td>...</td>
      <td>0.532</td>
      <td>1.000</td>
      <td>0.142</td>
      <td>0.083</td>
      <td>0.041</td>
      <td>0.062</td>
      <td>0.016</td>
      <td>0.037</td>
      <td>0.170</td>
      <td>0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1.000</td>
      <td>0.249</td>
      <td>0.075</td>
      <td>0.196</td>
      <td>0.165</td>
      <td>0.097</td>
      <td>0.075</td>
      <td>0.171</td>
      <td>0.100</td>
      <td>0.100</td>
      <td>...</td>
      <td>0.346</td>
      <td>0.406</td>
      <td>0.296</td>
      <td>0.294</td>
      <td>0.402</td>
      <td>0.346</td>
      <td>0.854</td>
      <td>0.913</td>
      <td>1.000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.450</td>
      <td>0.464</td>
      <td>0.630</td>
      <td>0.281</td>
      <td>0.267</td>
      <td>0.326</td>
      <td>0.722</td>
      <td>1.000</td>
      <td>0.575</td>
      <td>0.615</td>
      <td>...</td>
      <td>0.190</td>
      <td>0.027</td>
      <td>0.031</td>
      <td>0.276</td>
      <td>0.014</td>
      <td>0.015</td>
      <td>0.401</td>
      <td>0.036</td>
      <td>0.010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.724</td>
      <td>1.000</td>
      <td>0.423</td>
      <td>0.313</td>
      <td>0.395</td>
      <td>0.419</td>
      <td>0.525</td>
      <td>0.548</td>
      <td>0.506</td>
      <td>0.471</td>
      <td>...</td>
      <td>0.707</td>
      <td>0.855</td>
      <td>0.737</td>
      <td>0.505</td>
      <td>0.779</td>
      <td>0.398</td>
      <td>0.441</td>
      <td>0.541</td>
      <td>0.711</td>
      <td>0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1.000</td>
      <td>0.388</td>
      <td>0.038</td>
      <td>0.011</td>
      <td>0.065</td>
      <td>0.006</td>
      <td>0.006</td>
      <td>0.043</td>
      <td>0.008</td>
      <td>0.010</td>
      <td>...</td>
      <td>0.005</td>
      <td>0.006</td>
      <td>0.009</td>
      <td>0.245</td>
      <td>1.000</td>
      <td>0.193</td>
      <td>0.027</td>
      <td>0.017</td>
      <td>0.670</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.028</td>
      <td>0.040</td>
      <td>0.098</td>
      <td>0.566</td>
      <td>0.241</td>
      <td>0.109</td>
      <td>0.058</td>
      <td>0.061</td>
      <td>0.255</td>
      <td>1.000</td>
      <td>...</td>
      <td>0.030</td>
      <td>0.144</td>
      <td>0.316</td>
      <td>0.289</td>
      <td>0.302</td>
      <td>0.962</td>
      <td>1.000</td>
      <td>0.603</td>
      <td>0.167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.840</td>
      <td>1.000</td>
      <td>0.387</td>
      <td>0.124</td>
      <td>0.454</td>
      <td>0.443</td>
      <td>0.242</td>
      <td>0.117</td>
      <td>0.130</td>
      <td>0.096</td>
      <td>...</td>
      <td>0.188</td>
      <td>0.231</td>
      <td>0.332</td>
      <td>0.174</td>
      <td>0.242</td>
      <td>0.444</td>
      <td>0.229</td>
      <td>0.236</td>
      <td>0.188</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.344</td>
      <td>0.324</td>
      <td>0.552</td>
      <td>0.283</td>
      <td>0.171</td>
      <td>0.842</td>
      <td>0.210</td>
      <td>1.000</td>
      <td>0.894</td>
      <td>0.429</td>
      <td>...</td>
      <td>0.049</td>
      <td>0.012</td>
      <td>0.266</td>
      <td>1.000</td>
      <td>0.089</td>
      <td>0.224</td>
      <td>0.011</td>
      <td>0.019</td>
      <td>0.019</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1.000</td>
      <td>0.790</td>
      <td>0.425</td>
      <td>0.366</td>
      <td>0.456</td>
      <td>0.660</td>
      <td>0.462</td>
      <td>0.424</td>
      <td>0.113</td>
      <td>0.151</td>
      <td>...</td>
      <td>0.247</td>
      <td>0.824</td>
      <td>0.313</td>
      <td>0.222</td>
      <td>0.218</td>
      <td>0.247</td>
      <td>1.000</td>
      <td>0.512</td>
      <td>0.539</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.725</td>
      <td>1.000</td>
      <td>0.254</td>
      <td>0.536</td>
      <td>0.265</td>
      <td>0.168</td>
      <td>0.107</td>
      <td>0.194</td>
      <td>0.074</td>
      <td>0.070</td>
      <td>...</td>
      <td>0.696</td>
      <td>0.604</td>
      <td>0.422</td>
      <td>0.277</td>
      <td>1.000</td>
      <td>0.755</td>
      <td>0.374</td>
      <td>0.175</td>
      <td>0.163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.064</td>
      <td>0.057</td>
      <td>0.122</td>
      <td>0.108</td>
      <td>0.221</td>
      <td>0.591</td>
      <td>1.000</td>
      <td>0.929</td>
      <td>0.305</td>
      <td>0.108</td>
      <td>...</td>
      <td>0.162</td>
      <td>0.134</td>
      <td>0.119</td>
      <td>1.000</td>
      <td>0.240</td>
      <td>0.084</td>
      <td>0.214</td>
      <td>0.214</td>
      <td>0.337</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.462</td>
      <td>0.332</td>
      <td>0.322</td>
      <td>0.345</td>
      <td>0.396</td>
      <td>1.000</td>
      <td>0.616</td>
      <td>0.673</td>
      <td>0.644</td>
      <td>0.378</td>
      <td>...</td>
      <td>0.201</td>
      <td>0.363</td>
      <td>1.000</td>
      <td>0.403</td>
      <td>0.398</td>
      <td>0.333</td>
      <td>0.207</td>
      <td>0.143</td>
      <td>0.158</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>1.000</td>
      <td>0.356</td>
      <td>0.079</td>
      <td>0.028</td>
      <td>0.020</td>
      <td>0.107</td>
      <td>0.079</td>
      <td>0.060</td>
      <td>0.142</td>
      <td>0.016</td>
      <td>...</td>
      <td>0.228</td>
      <td>0.294</td>
      <td>1.000</td>
      <td>0.900</td>
      <td>0.145</td>
      <td>0.140</td>
      <td>0.172</td>
      <td>0.150</td>
      <td>0.081</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.858</td>
      <td>0.445</td>
      <td>0.178</td>
      <td>0.148</td>
      <td>0.494</td>
      <td>0.213</td>
      <td>0.269</td>
      <td>0.396</td>
      <td>0.841</td>
      <td>0.644</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.999</td>
      <td>0.841</td>
      <td>0.621</td>
      <td>0.396</td>
      <td>0.411</td>
      <td>0.283</td>
      <td>0.226</td>
      <td>0.606</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.168</td>
      <td>0.199</td>
      <td>0.211</td>
      <td>1.000</td>
      <td>0.386</td>
      <td>0.219</td>
      <td>0.187</td>
      <td>0.312</td>
      <td>0.196</td>
      <td>0.116</td>
      <td>...</td>
      <td>0.137</td>
      <td>0.107</td>
      <td>0.191</td>
      <td>0.144</td>
      <td>0.311</td>
      <td>0.090</td>
      <td>0.057</td>
      <td>0.138</td>
      <td>0.501</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.378</td>
      <td>0.198</td>
      <td>0.457</td>
      <td>0.274</td>
      <td>0.213</td>
      <td>0.241</td>
      <td>0.246</td>
      <td>1.000</td>
      <td>0.224</td>
      <td>0.360</td>
      <td>...</td>
      <td>0.366</td>
      <td>0.331</td>
      <td>0.341</td>
      <td>0.428</td>
      <td>1.000</td>
      <td>0.358</td>
      <td>0.355</td>
      <td>0.122</td>
      <td>0.167</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1.000</td>
      <td>0.301</td>
      <td>0.052</td>
      <td>0.037</td>
      <td>0.147</td>
      <td>0.019</td>
      <td>0.059</td>
      <td>0.027</td>
      <td>0.231</td>
      <td>0.574</td>
      <td>...</td>
      <td>0.269</td>
      <td>0.373</td>
      <td>0.222</td>
      <td>0.628</td>
      <td>0.825</td>
      <td>0.765</td>
      <td>0.304</td>
      <td>0.225</td>
      <td>0.517</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.089</td>
      <td>0.748</td>
      <td>0.175</td>
      <td>1.000</td>
      <td>0.193</td>
      <td>0.189</td>
      <td>0.058</td>
      <td>0.127</td>
      <td>0.315</td>
      <td>0.051</td>
      <td>...</td>
      <td>1.000</td>
      <td>0.217</td>
      <td>0.079</td>
      <td>0.042</td>
      <td>0.116</td>
      <td>0.249</td>
      <td>0.133</td>
      <td>0.247</td>
      <td>0.311</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1.000</td>
      <td>0.689</td>
      <td>0.656</td>
      <td>0.813</td>
      <td>0.727</td>
      <td>0.568</td>
      <td>0.445</td>
      <td>0.469</td>
      <td>0.434</td>
      <td>0.301</td>
      <td>...</td>
      <td>0.542</td>
      <td>0.478</td>
      <td>0.409</td>
      <td>0.417</td>
      <td>0.473</td>
      <td>0.378</td>
      <td>0.380</td>
      <td>0.224</td>
      <td>0.989</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.028</td>
      <td>0.107</td>
      <td>0.042</td>
      <td>0.037</td>
      <td>0.099</td>
      <td>0.153</td>
      <td>1.000</td>
      <td>0.140</td>
      <td>0.074</td>
      <td>0.336</td>
      <td>...</td>
      <td>0.026</td>
      <td>0.048</td>
      <td>0.273</td>
      <td>1.000</td>
      <td>0.074</td>
      <td>0.024</td>
      <td>0.218</td>
      <td>0.024</td>
      <td>0.032</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.211</td>
      <td>0.448</td>
      <td>0.396</td>
      <td>0.399</td>
      <td>0.902</td>
      <td>0.900</td>
      <td>1.000</td>
      <td>0.946</td>
      <td>0.604</td>
      <td>0.645</td>
      <td>...</td>
      <td>0.272</td>
      <td>0.478</td>
      <td>0.468</td>
      <td>0.478</td>
      <td>0.456</td>
      <td>0.176</td>
      <td>0.180</td>
      <td>0.186</td>
      <td>0.124</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.450</td>
      <td>0.387</td>
      <td>1.000</td>
      <td>0.552</td>
      <td>0.740</td>
      <td>0.163</td>
      <td>0.176</td>
      <td>0.221</td>
      <td>0.076</td>
      <td>0.202</td>
      <td>...</td>
      <td>0.538</td>
      <td>0.771</td>
      <td>0.486</td>
      <td>1.000</td>
      <td>0.724</td>
      <td>0.554</td>
      <td>0.388</td>
      <td>0.264</td>
      <td>0.322</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.199</td>
      <td>1.000</td>
      <td>0.219</td>
      <td>0.045</td>
      <td>0.009</td>
      <td>0.032</td>
      <td>0.018</td>
      <td>0.028</td>
      <td>0.121</td>
      <td>0.029</td>
      <td>...</td>
      <td>0.158</td>
      <td>0.105</td>
      <td>0.069</td>
      <td>0.231</td>
      <td>0.083</td>
      <td>0.051</td>
      <td>0.050</td>
      <td>0.093</td>
      <td>0.540</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.099</td>
      <td>0.046</td>
      <td>0.008</td>
      <td>0.007</td>
      <td>0.069</td>
      <td>0.030</td>
      <td>0.306</td>
      <td>0.303</td>
      <td>0.969</td>
      <td>1.000</td>
      <td>...</td>
      <td>0.547</td>
      <td>0.490</td>
      <td>0.063</td>
      <td>0.030</td>
      <td>0.033</td>
      <td>0.044</td>
      <td>0.039</td>
      <td>0.024</td>
      <td>0.057</td>
      <td>0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.489</td>
      <td>1.000</td>
      <td>0.150</td>
      <td>0.114</td>
      <td>0.290</td>
      <td>0.126</td>
      <td>0.130</td>
      <td>0.134</td>
      <td>0.235</td>
      <td>0.970</td>
      <td>...</td>
      <td>0.289</td>
      <td>1.000</td>
      <td>0.203</td>
      <td>0.214</td>
      <td>0.069</td>
      <td>0.250</td>
      <td>0.092</td>
      <td>0.075</td>
      <td>0.567</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.137</td>
      <td>0.142</td>
      <td>0.259</td>
      <td>0.195</td>
      <td>0.366</td>
      <td>0.563</td>
      <td>0.515</td>
      <td>1.000</td>
      <td>0.218</td>
      <td>0.185</td>
      <td>...</td>
      <td>0.344</td>
      <td>0.188</td>
      <td>0.505</td>
      <td>0.557</td>
      <td>0.670</td>
      <td>0.651</td>
      <td>0.275</td>
      <td>0.249</td>
      <td>0.671</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.220</td>
      <td>1.000</td>
      <td>0.637</td>
      <td>0.590</td>
      <td>0.319</td>
      <td>0.310</td>
      <td>0.338</td>
      <td>0.167</td>
      <td>0.029</td>
      <td>0.065</td>
      <td>...</td>
      <td>0.933</td>
      <td>0.661</td>
      <td>0.654</td>
      <td>0.298</td>
      <td>0.318</td>
      <td>0.190</td>
      <td>0.204</td>
      <td>0.632</td>
      <td>0.161</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.037</td>
      <td>0.195</td>
      <td>0.042</td>
      <td>0.048</td>
      <td>0.306</td>
      <td>0.050</td>
      <td>0.423</td>
      <td>0.138</td>
      <td>0.128</td>
      <td>1.000</td>
      <td>...</td>
      <td>0.066</td>
      <td>0.113</td>
      <td>0.161</td>
      <td>0.308</td>
      <td>0.137</td>
      <td>0.530</td>
      <td>1.000</td>
      <td>0.403</td>
      <td>0.326</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.900</td>
      <td>0.670</td>
      <td>0.480</td>
      <td>0.415</td>
      <td>0.625</td>
      <td>0.771</td>
      <td>0.856</td>
      <td>0.732</td>
      <td>0.890</td>
      <td>0.898</td>
      <td>...</td>
      <td>0.290</td>
      <td>0.050</td>
      <td>0.104</td>
      <td>0.035</td>
      <td>0.071</td>
      <td>1.000</td>
      <td>0.088</td>
      <td>0.068</td>
      <td>0.019</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.947</td>
      <td>1.000</td>
      <td>0.714</td>
      <td>0.422</td>
      <td>0.374</td>
      <td>0.581</td>
      <td>0.595</td>
      <td>0.457</td>
      <td>0.386</td>
      <td>0.352</td>
      <td>...</td>
      <td>0.117</td>
      <td>0.575</td>
      <td>0.182</td>
      <td>0.261</td>
      <td>0.091</td>
      <td>0.344</td>
      <td>0.079</td>
      <td>0.177</td>
      <td>0.345</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>79 rows × 217 columns</p>
</div>




```python
feature_cols = [x for x in range(216)]

#grab all rows, and all cols
Xtrain = df_alltrain.loc[:, feature_cols]
Xtest = df_alltest.loc[:, feature_cols]

# y response value in training set is always true, for 39 songs
ytrain = np.ravel(np.array([[x] for x in df_alltrain['whoop']]))
ytest = np.ravel(np.array([[x] for x in df_alltest['whoop']]))

clf = SVC(kernel='linear', C=1.0)
#xtrain = clf.fit(X,y)
#test_size =100
clf.fit(Xtrain,ytrain)
outcome = clf.predict(Xtest)
print(metrics.accuracy_score(ytest, outcome))

print(ytest)
print(outcome)
```

    0.4050632911392405
    ['1' '1' '1' '1' '1' '1' '1' '1' '1' '1' '0' '0' '0' '0' '0' '0' '0' '0'
     '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0'
     '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0'
     '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0'
     '0' '0' '0' '0' '0' '0' '0']
    ['1' '0' '0' '0' '0' '1' '0' '1' '0' '0' '1' '1' '1' '1' '0' '1' '0' '1'
     '0' '1' '0' '1' '1' '1' '0' '1' '0' '1' '1' '0' '0' '1' '1' '0' '1' '1'
     '1' '0' '0' '0' '0' '0' '1' '0' '0' '0' '1' '1' '1' '1' '0' '1' '1' '1'
     '0' '1' '1' '0' '1' '0' '1' '0' '1' '0' '1' '0' '0' '1' '1' '0' '0' '1'
     '1' '0' '1' '1' '0' '1' '1']
    

**Outcome of Model 1:** The above model that is trained on mostly class 1 y values (whoop=True) performs very poorly when tested on data which contains a small percentage of class 1 values.

**Question for Model 2:** What if we first shuffle the class 1 and class 0 values before splitting the training and test data?


```python
#Question: Does our model perform better if we split the data arbitrarily?
#shuffle index at random for model
from sklearn.utils import shuffle
df_pitches['whoop']= '1'
df_pitchesnon['whoop'] = '0'
df_all2 = pd.concat([df_pitches, df_pitchesnon])
df_all2
df_all2 = shuffle(df_all2)
```


```python
feature_cols = [x for x in range(216)]

#grab all rows, and all cols
X = df_all2.loc[:, feature_cols]

# y response value in training set is always true, for 39 songs
y = np.ravel(np.array([[x] for x in df_all2['whoop']]))

clf = SVC(kernel='rbf', C=1.0, gamma='auto')
xtrain = clf.fit(X,y)
test_size =100
clf.fit(X[:test_size],y[:test_size])
outcome = clf.predict(X[test_size:])
print(metrics.accuracy_score(y[test_size:], outcome))

print(y[test_size:])
print(outcome)
```

    0.6666666666666666
    ['0' '0' '0' '1' '0' '0' '1' '0' '0' '1' '1' '1' '0' '0' '1' '1' '0' '1'
     '1' '0' '1' '0' '0' '1' '0' '0' '0' '0' '1' '0' '0' '0' '1' '0' '0' '0'
     '0' '0' '0']
    ['0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0'
     '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0' '0'
     '0' '0' '0']
    

**Outcome for Model 2:** Our model is now performing better, however it is not performing any better 
than a random prediction algorithm might. 

**Proposal:** To improve the performance of our SVC model, we will consider finding a similarity 
measure to understand the difference between pitches accross time in two songs that contain a "whoop" 
versus those that do not contain a whoop. 

Naive approach = Euclidean Distance measure of similarity


```python
#Compute Euclidean Distance (where euc distance between two points in 2D space = abs(x1-x2) for each xn ->|xj- avg(x0..xn))

# Create 1*n array where n=number pithces in entire "whoop" segment, and each value is the avg x val (aggregate) for that pithc, pitch j
# for each col in df, calculate avg over col and collapse into 1d array
means = df_pitches.drop('whoop', 1).agg(["mean"])

# map lambda function onto df_pitches that calcs for each value x the euc distance x from the avg(x) for that pitch#
for i in range(len(means.index)):
    df_eucDist = df_all2.drop('whoop',1).apply(lambda x: abs(x-means[i][0])*2)

#df_eucDist
#df_all2

#we can now append the "whoop" column back on from the original df_all2
#print(df_eucDist.join(df_all2[df_all2.columns[-1]]))
df_eucDist['whoop']=df_all2['whoop'].astype(int)
df_eucDist


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>207</th>
      <th>208</th>
      <th>209</th>
      <th>210</th>
      <th>211</th>
      <th>212</th>
      <th>213</th>
      <th>214</th>
      <th>215</th>
      <th>whoop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0.125077</td>
      <td>0.364923</td>
      <td>1.080923</td>
      <td>0.578923</td>
      <td>0.240923</td>
      <td>0.822923</td>
      <td>0.203077</td>
      <td>0.491077</td>
      <td>0.625077</td>
      <td>0.501077</td>
      <td>...</td>
      <td>0.805077</td>
      <td>0.389077</td>
      <td>1.080923</td>
      <td>0.311077</td>
      <td>0.799077</td>
      <td>0.843077</td>
      <td>0.691077</td>
      <td>0.813077</td>
      <td>0.747077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.739077</td>
      <td>0.502923</td>
      <td>0.713077</td>
      <td>0.074923</td>
      <td>0.265077</td>
      <td>0.811077</td>
      <td>0.735077</td>
      <td>0.711077</td>
      <td>1.080923</td>
      <td>0.755077</td>
      <td>...</td>
      <td>0.579077</td>
      <td>0.632923</td>
      <td>0.659077</td>
      <td>0.242923</td>
      <td>0.509077</td>
      <td>0.231077</td>
      <td>0.693077</td>
      <td>0.739077</td>
      <td>1.080923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.266923</td>
      <td>1.080923</td>
      <td>0.722923</td>
      <td>0.416923</td>
      <td>0.480923</td>
      <td>0.076923</td>
      <td>0.496923</td>
      <td>0.127077</td>
      <td>0.255077</td>
      <td>0.140923</td>
      <td>...</td>
      <td>0.553077</td>
      <td>0.421077</td>
      <td>0.601077</td>
      <td>0.615077</td>
      <td>0.653077</td>
      <td>0.417077</td>
      <td>0.391077</td>
      <td>0.575077</td>
      <td>0.825077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.528923</td>
      <td>1.080923</td>
      <td>0.073077</td>
      <td>0.293077</td>
      <td>0.129077</td>
      <td>0.081077</td>
      <td>0.130923</td>
      <td>0.176923</td>
      <td>0.092923</td>
      <td>0.022923</td>
      <td>...</td>
      <td>0.494923</td>
      <td>0.790923</td>
      <td>0.554923</td>
      <td>0.090923</td>
      <td>0.638923</td>
      <td>0.123077</td>
      <td>0.037077</td>
      <td>0.162923</td>
      <td>0.502923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0.739077</td>
      <td>0.751077</td>
      <td>0.795077</td>
      <td>0.309077</td>
      <td>1.080923</td>
      <td>0.823077</td>
      <td>0.863077</td>
      <td>0.627077</td>
      <td>0.663077</td>
      <td>0.861077</td>
      <td>...</td>
      <td>0.653077</td>
      <td>0.653077</td>
      <td>0.371077</td>
      <td>0.605077</td>
      <td>0.813077</td>
      <td>0.687077</td>
      <td>0.557077</td>
      <td>0.671077</td>
      <td>0.535077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.606923</td>
      <td>1.080923</td>
      <td>0.689077</td>
      <td>0.625077</td>
      <td>0.611077</td>
      <td>0.613077</td>
      <td>0.577077</td>
      <td>0.080923</td>
      <td>0.480923</td>
      <td>0.375077</td>
      <td>...</td>
      <td>0.361077</td>
      <td>0.739077</td>
      <td>0.449077</td>
      <td>0.829077</td>
      <td>0.849077</td>
      <td>0.819077</td>
      <td>0.667077</td>
      <td>1.080923</td>
      <td>0.687077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>0.691077</td>
      <td>0.042923</td>
      <td>1.080923</td>
      <td>0.131077</td>
      <td>0.671077</td>
      <td>0.783077</td>
      <td>0.745077</td>
      <td>0.687077</td>
      <td>0.845077</td>
      <td>0.825077</td>
      <td>...</td>
      <td>0.601077</td>
      <td>0.481077</td>
      <td>0.361077</td>
      <td>0.627077</td>
      <td>0.773077</td>
      <td>0.731077</td>
      <td>0.809077</td>
      <td>0.793077</td>
      <td>0.970923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.163077</td>
      <td>0.523077</td>
      <td>0.005077</td>
      <td>0.371077</td>
      <td>0.493077</td>
      <td>0.437077</td>
      <td>0.427077</td>
      <td>1.080923</td>
      <td>0.471077</td>
      <td>0.199077</td>
      <td>...</td>
      <td>0.187077</td>
      <td>0.257077</td>
      <td>0.237077</td>
      <td>0.063077</td>
      <td>1.080923</td>
      <td>0.203077</td>
      <td>0.209077</td>
      <td>0.675077</td>
      <td>0.585077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.685077</td>
      <td>0.599077</td>
      <td>0.751077</td>
      <td>0.745077</td>
      <td>0.537077</td>
      <td>0.727077</td>
      <td>0.635077</td>
      <td>0.669077</td>
      <td>0.471077</td>
      <td>1.080923</td>
      <td>...</td>
      <td>0.463077</td>
      <td>1.080923</td>
      <td>0.401077</td>
      <td>0.439077</td>
      <td>0.781077</td>
      <td>0.823077</td>
      <td>0.789077</td>
      <td>0.805077</td>
      <td>0.615077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.845077</td>
      <td>0.529077</td>
      <td>0.835077</td>
      <td>0.823077</td>
      <td>0.307077</td>
      <td>0.819077</td>
      <td>0.073077</td>
      <td>0.643077</td>
      <td>0.663077</td>
      <td>1.080923</td>
      <td>...</td>
      <td>0.787077</td>
      <td>0.693077</td>
      <td>0.597077</td>
      <td>0.303077</td>
      <td>0.645077</td>
      <td>0.140923</td>
      <td>1.080923</td>
      <td>0.113077</td>
      <td>0.267077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.619077</td>
      <td>0.155077</td>
      <td>1.080923</td>
      <td>0.301077</td>
      <td>0.761077</td>
      <td>0.547077</td>
      <td>0.548923</td>
      <td>1.024923</td>
      <td>0.291077</td>
      <td>0.633077</td>
      <td>...</td>
      <td>0.620923</td>
      <td>0.288923</td>
      <td>0.411077</td>
      <td>0.029077</td>
      <td>0.401077</td>
      <td>0.603077</td>
      <td>0.513077</td>
      <td>0.655077</td>
      <td>0.079077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.080923</td>
      <td>0.690923</td>
      <td>0.219077</td>
      <td>0.229077</td>
      <td>0.173077</td>
      <td>0.295077</td>
      <td>0.121077</td>
      <td>0.191077</td>
      <td>0.582923</td>
      <td>0.100923</td>
      <td>...</td>
      <td>0.610923</td>
      <td>1.080923</td>
      <td>0.119077</td>
      <td>0.072923</td>
      <td>0.479077</td>
      <td>0.275077</td>
      <td>0.796923</td>
      <td>0.303077</td>
      <td>0.071077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.203077</td>
      <td>0.121077</td>
      <td>0.142923</td>
      <td>0.688923</td>
      <td>0.425077</td>
      <td>0.085077</td>
      <td>1.080923</td>
      <td>0.419077</td>
      <td>0.315077</td>
      <td>0.219077</td>
      <td>...</td>
      <td>0.153077</td>
      <td>0.297077</td>
      <td>0.523077</td>
      <td>0.291077</td>
      <td>0.194923</td>
      <td>1.080923</td>
      <td>0.315077</td>
      <td>0.565077</td>
      <td>0.651077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1.076923</td>
      <td>0.501077</td>
      <td>0.599077</td>
      <td>0.046923</td>
      <td>0.649077</td>
      <td>1.080923</td>
      <td>0.495077</td>
      <td>0.427077</td>
      <td>0.299077</td>
      <td>0.447077</td>
      <td>...</td>
      <td>1.080923</td>
      <td>0.747077</td>
      <td>0.759077</td>
      <td>0.809077</td>
      <td>0.461077</td>
      <td>0.835077</td>
      <td>0.783077</td>
      <td>0.327077</td>
      <td>0.667077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.603077</td>
      <td>0.681077</td>
      <td>0.755077</td>
      <td>0.639077</td>
      <td>0.655077</td>
      <td>0.717077</td>
      <td>0.783077</td>
      <td>1.080923</td>
      <td>0.479077</td>
      <td>0.829077</td>
      <td>...</td>
      <td>0.733077</td>
      <td>0.541077</td>
      <td>1.080923</td>
      <td>0.518923</td>
      <td>0.693077</td>
      <td>0.787077</td>
      <td>0.605077</td>
      <td>0.853077</td>
      <td>0.837077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.755077</td>
      <td>0.803077</td>
      <td>0.721077</td>
      <td>0.215077</td>
      <td>0.773077</td>
      <td>0.777077</td>
      <td>0.851077</td>
      <td>0.643077</td>
      <td>0.759077</td>
      <td>0.681077</td>
      <td>...</td>
      <td>1.080923</td>
      <td>0.344923</td>
      <td>0.367077</td>
      <td>0.110923</td>
      <td>0.050923</td>
      <td>0.589077</td>
      <td>0.058923</td>
      <td>0.338923</td>
      <td>0.275077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.188923</td>
      <td>1.080923</td>
      <td>0.938923</td>
      <td>0.684923</td>
      <td>0.311077</td>
      <td>0.271077</td>
      <td>0.649077</td>
      <td>0.745077</td>
      <td>0.735077</td>
      <td>0.709077</td>
      <td>...</td>
      <td>0.768923</td>
      <td>1.022923</td>
      <td>1.080923</td>
      <td>0.816923</td>
      <td>0.812923</td>
      <td>0.452923</td>
      <td>0.460923</td>
      <td>0.311077</td>
      <td>0.299077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1.080923</td>
      <td>0.590923</td>
      <td>0.429077</td>
      <td>0.641077</td>
      <td>0.571077</td>
      <td>0.589077</td>
      <td>0.449077</td>
      <td>0.113077</td>
      <td>0.062923</td>
      <td>0.059077</td>
      <td>...</td>
      <td>0.211077</td>
      <td>0.113077</td>
      <td>0.383077</td>
      <td>0.416923</td>
      <td>1.080923</td>
      <td>0.073077</td>
      <td>0.076923</td>
      <td>0.333077</td>
      <td>0.960923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.176923</td>
      <td>1.080923</td>
      <td>0.663077</td>
      <td>0.629077</td>
      <td>0.425077</td>
      <td>0.781077</td>
      <td>0.695077</td>
      <td>0.695077</td>
      <td>0.382923</td>
      <td>0.845077</td>
      <td>...</td>
      <td>0.365077</td>
      <td>1.080923</td>
      <td>0.379077</td>
      <td>0.599077</td>
      <td>0.699077</td>
      <td>0.561077</td>
      <td>0.755077</td>
      <td>0.681077</td>
      <td>0.009077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>0.122923</td>
      <td>1.080923</td>
      <td>0.423077</td>
      <td>0.837077</td>
      <td>0.813077</td>
      <td>0.843077</td>
      <td>0.881077</td>
      <td>0.771077</td>
      <td>0.319077</td>
      <td>0.407077</td>
      <td>...</td>
      <td>0.761077</td>
      <td>0.743077</td>
      <td>0.557077</td>
      <td>1.080923</td>
      <td>0.783077</td>
      <td>0.685077</td>
      <td>0.233077</td>
      <td>0.741077</td>
      <td>0.767077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.007077</td>
      <td>1.024923</td>
      <td>1.080923</td>
      <td>0.972923</td>
      <td>1.006923</td>
      <td>1.004923</td>
      <td>0.706923</td>
      <td>0.710923</td>
      <td>0.630923</td>
      <td>0.634923</td>
      <td>...</td>
      <td>0.446923</td>
      <td>0.266923</td>
      <td>0.302923</td>
      <td>0.266923</td>
      <td>0.234923</td>
      <td>0.086923</td>
      <td>0.039077</td>
      <td>0.089077</td>
      <td>1.080923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.695077</td>
      <td>0.823077</td>
      <td>0.835077</td>
      <td>0.787077</td>
      <td>0.495077</td>
      <td>1.080923</td>
      <td>0.235077</td>
      <td>0.885077</td>
      <td>0.879077</td>
      <td>0.821077</td>
      <td>...</td>
      <td>0.875077</td>
      <td>0.423077</td>
      <td>1.080923</td>
      <td>0.629077</td>
      <td>0.335077</td>
      <td>0.887077</td>
      <td>0.871077</td>
      <td>0.911077</td>
      <td>0.853077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.679077</td>
      <td>0.171077</td>
      <td>0.249077</td>
      <td>0.207077</td>
      <td>0.828923</td>
      <td>0.698923</td>
      <td>1.080923</td>
      <td>0.812923</td>
      <td>0.089077</td>
      <td>0.069077</td>
      <td>...</td>
      <td>0.457077</td>
      <td>0.612923</td>
      <td>0.032923</td>
      <td>0.990923</td>
      <td>1.080923</td>
      <td>0.134923</td>
      <td>0.306923</td>
      <td>0.193077</td>
      <td>0.283077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1.080923</td>
      <td>0.143077</td>
      <td>0.843077</td>
      <td>0.897077</td>
      <td>0.789077</td>
      <td>0.907077</td>
      <td>0.907077</td>
      <td>0.833077</td>
      <td>0.903077</td>
      <td>0.899077</td>
      <td>...</td>
      <td>0.909077</td>
      <td>0.907077</td>
      <td>0.901077</td>
      <td>0.429077</td>
      <td>1.080923</td>
      <td>0.533077</td>
      <td>0.865077</td>
      <td>0.885077</td>
      <td>0.420923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.586923</td>
      <td>0.622923</td>
      <td>0.430923</td>
      <td>0.256923</td>
      <td>1.022923</td>
      <td>0.898923</td>
      <td>0.852923</td>
      <td>1.080923</td>
      <td>0.063077</td>
      <td>0.071077</td>
      <td>...</td>
      <td>0.367077</td>
      <td>0.454923</td>
      <td>0.278923</td>
      <td>0.794923</td>
      <td>1.080923</td>
      <td>0.088923</td>
      <td>0.076923</td>
      <td>0.767077</td>
      <td>0.851077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.741077</td>
      <td>0.576923</td>
      <td>0.569077</td>
      <td>1.080923</td>
      <td>0.533077</td>
      <td>0.541077</td>
      <td>0.803077</td>
      <td>0.665077</td>
      <td>0.289077</td>
      <td>0.817077</td>
      <td>...</td>
      <td>1.080923</td>
      <td>0.485077</td>
      <td>0.761077</td>
      <td>0.835077</td>
      <td>0.687077</td>
      <td>0.421077</td>
      <td>0.653077</td>
      <td>0.425077</td>
      <td>0.297077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.341077</td>
      <td>0.534923</td>
      <td>0.508923</td>
      <td>0.604923</td>
      <td>1.080923</td>
      <td>1.054923</td>
      <td>0.530923</td>
      <td>0.540923</td>
      <td>0.431077</td>
      <td>0.445077</td>
      <td>...</td>
      <td>0.637077</td>
      <td>0.948923</td>
      <td>1.080923</td>
      <td>0.257077</td>
      <td>0.515077</td>
      <td>0.601077</td>
      <td>0.807077</td>
      <td>0.793077</td>
      <td>0.235077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.080923</td>
      <td>0.689077</td>
      <td>0.807077</td>
      <td>0.787077</td>
      <td>0.689077</td>
      <td>0.614923</td>
      <td>0.735077</td>
      <td>0.603077</td>
      <td>0.837077</td>
      <td>0.387077</td>
      <td>...</td>
      <td>0.116923</td>
      <td>0.156923</td>
      <td>0.114923</td>
      <td>0.096923</td>
      <td>0.384923</td>
      <td>0.131077</td>
      <td>0.219077</td>
      <td>0.363077</td>
      <td>0.870923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.080923</td>
      <td>0.373077</td>
      <td>0.539077</td>
      <td>0.481077</td>
      <td>0.181077</td>
      <td>1.034923</td>
      <td>0.591077</td>
      <td>0.481077</td>
      <td>0.197077</td>
      <td>0.635077</td>
      <td>...</td>
      <td>1.080923</td>
      <td>0.135077</td>
      <td>0.147077</td>
      <td>0.557077</td>
      <td>0.503077</td>
      <td>0.121077</td>
      <td>0.083077</td>
      <td>0.754923</td>
      <td>0.329077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.837077</td>
      <td>0.012923</td>
      <td>1.080923</td>
      <td>0.009077</td>
      <td>0.707077</td>
      <td>0.863077</td>
      <td>0.873077</td>
      <td>0.883077</td>
      <td>0.865077</td>
      <td>0.527077</td>
      <td>...</td>
      <td>0.509077</td>
      <td>0.012923</td>
      <td>0.014923</td>
      <td>0.646923</td>
      <td>0.586923</td>
      <td>1.052923</td>
      <td>1.080923</td>
      <td>1.056923</td>
      <td>0.867077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.679077</td>
      <td>0.877077</td>
      <td>0.683077</td>
      <td>0.883077</td>
      <td>0.837077</td>
      <td>0.761077</td>
      <td>0.717077</td>
      <td>0.877077</td>
      <td>0.839077</td>
      <td>0.425077</td>
      <td>...</td>
      <td>0.721077</td>
      <td>0.817077</td>
      <td>0.743077</td>
      <td>0.671077</td>
      <td>0.239077</td>
      <td>0.775077</td>
      <td>0.667077</td>
      <td>0.867077</td>
      <td>0.529077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.809077</td>
      <td>0.527077</td>
      <td>0.829077</td>
      <td>0.787077</td>
      <td>0.663077</td>
      <td>0.403077</td>
      <td>1.080923</td>
      <td>0.783077</td>
      <td>0.793077</td>
      <td>0.825077</td>
      <td>...</td>
      <td>0.407077</td>
      <td>0.665077</td>
      <td>0.621077</td>
      <td>0.661077</td>
      <td>0.535077</td>
      <td>1.080923</td>
      <td>0.455077</td>
      <td>0.497077</td>
      <td>0.593077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.127077</td>
      <td>1.080923</td>
      <td>0.439077</td>
      <td>0.663077</td>
      <td>0.751077</td>
      <td>0.665077</td>
      <td>0.201077</td>
      <td>0.427077</td>
      <td>0.209077</td>
      <td>0.155077</td>
      <td>...</td>
      <td>0.245077</td>
      <td>0.484923</td>
      <td>0.788923</td>
      <td>1.080923</td>
      <td>0.700923</td>
      <td>0.009077</td>
      <td>0.189077</td>
      <td>0.519077</td>
      <td>0.453077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.791077</td>
      <td>0.805077</td>
      <td>0.675077</td>
      <td>0.703077</td>
      <td>0.477077</td>
      <td>0.262923</td>
      <td>1.080923</td>
      <td>0.938923</td>
      <td>0.309077</td>
      <td>0.703077</td>
      <td>...</td>
      <td>0.595077</td>
      <td>0.651077</td>
      <td>0.681077</td>
      <td>1.080923</td>
      <td>0.439077</td>
      <td>0.751077</td>
      <td>0.491077</td>
      <td>0.491077</td>
      <td>0.245077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.467077</td>
      <td>0.599077</td>
      <td>0.597077</td>
      <td>0.433077</td>
      <td>0.000923</td>
      <td>0.245077</td>
      <td>0.423077</td>
      <td>1.080923</td>
      <td>0.213077</td>
      <td>0.695077</td>
      <td>...</td>
      <td>0.124923</td>
      <td>0.201077</td>
      <td>0.451077</td>
      <td>0.036923</td>
      <td>0.342923</td>
      <td>0.357077</td>
      <td>0.429077</td>
      <td>0.541077</td>
      <td>1.080923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.457077</td>
      <td>0.369077</td>
      <td>0.358923</td>
      <td>0.665077</td>
      <td>0.789077</td>
      <td>0.761077</td>
      <td>0.199077</td>
      <td>1.080923</td>
      <td>0.713077</td>
      <td>0.465077</td>
      <td>...</td>
      <td>0.797077</td>
      <td>0.469077</td>
      <td>0.777077</td>
      <td>0.555077</td>
      <td>1.080923</td>
      <td>0.681077</td>
      <td>0.873077</td>
      <td>0.815077</td>
      <td>0.461077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.320923</td>
      <td>0.291077</td>
      <td>0.235077</td>
      <td>0.475077</td>
      <td>0.192923</td>
      <td>0.512923</td>
      <td>0.026923</td>
      <td>1.080923</td>
      <td>0.281077</td>
      <td>0.019077</td>
      <td>...</td>
      <td>0.711077</td>
      <td>0.375077</td>
      <td>1.080923</td>
      <td>0.233077</td>
      <td>0.401077</td>
      <td>0.563077</td>
      <td>0.049077</td>
      <td>0.621077</td>
      <td>0.641077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1.080923</td>
      <td>0.344923</td>
      <td>0.387077</td>
      <td>0.593077</td>
      <td>0.619077</td>
      <td>0.571077</td>
      <td>0.401077</td>
      <td>0.001077</td>
      <td>0.397077</td>
      <td>0.639077</td>
      <td>...</td>
      <td>0.385077</td>
      <td>0.359077</td>
      <td>0.535077</td>
      <td>0.569077</td>
      <td>0.639077</td>
      <td>0.683077</td>
      <td>0.162923</td>
      <td>0.225077</td>
      <td>1.044923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.893077</td>
      <td>0.887077</td>
      <td>0.371077</td>
      <td>0.873077</td>
      <td>0.889077</td>
      <td>0.893077</td>
      <td>0.783077</td>
      <td>1.080923</td>
      <td>0.837077</td>
      <td>0.739077</td>
      <td>...</td>
      <td>0.809077</td>
      <td>0.493077</td>
      <td>1.080923</td>
      <td>0.161077</td>
      <td>0.697077</td>
      <td>0.669077</td>
      <td>0.671077</td>
      <td>0.791077</td>
      <td>0.707077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.530923</td>
      <td>1.080923</td>
      <td>0.154923</td>
      <td>0.422923</td>
      <td>0.504923</td>
      <td>0.059077</td>
      <td>0.070923</td>
      <td>0.323077</td>
      <td>0.212923</td>
      <td>0.288923</td>
      <td>...</td>
      <td>0.065077</td>
      <td>0.082923</td>
      <td>0.151077</td>
      <td>0.637077</td>
      <td>0.471077</td>
      <td>0.453077</td>
      <td>0.157077</td>
      <td>0.353077</td>
      <td>0.449077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.178923</td>
      <td>0.822923</td>
      <td>0.482923</td>
      <td>0.021077</td>
      <td>0.459077</td>
      <td>0.638923</td>
      <td>0.165077</td>
      <td>1.080923</td>
      <td>0.145077</td>
      <td>0.215077</td>
      <td>...</td>
      <td>0.619077</td>
      <td>0.767077</td>
      <td>0.657077</td>
      <td>0.607077</td>
      <td>0.775077</td>
      <td>0.805077</td>
      <td>0.399077</td>
      <td>0.565077</td>
      <td>0.787077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.289077</td>
      <td>0.092923</td>
      <td>0.225077</td>
      <td>0.495077</td>
      <td>0.417077</td>
      <td>0.218923</td>
      <td>1.080923</td>
      <td>1.006923</td>
      <td>0.508923</td>
      <td>0.284923</td>
      <td>...</td>
      <td>0.023077</td>
      <td>1.080923</td>
      <td>0.431077</td>
      <td>0.527077</td>
      <td>0.521077</td>
      <td>0.429077</td>
      <td>0.465077</td>
      <td>0.501077</td>
      <td>0.133077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.710923</td>
      <td>0.094923</td>
      <td>0.074923</td>
      <td>0.045077</td>
      <td>0.447077</td>
      <td>0.737077</td>
      <td>0.713077</td>
      <td>0.491077</td>
      <td>0.659077</td>
      <td>0.607077</td>
      <td>...</td>
      <td>1.016923</td>
      <td>0.170923</td>
      <td>0.343077</td>
      <td>0.433077</td>
      <td>0.339077</td>
      <td>0.181077</td>
      <td>0.283077</td>
      <td>0.327077</td>
      <td>0.043077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.454923</td>
      <td>0.582923</td>
      <td>0.748923</td>
      <td>0.097077</td>
      <td>0.457077</td>
      <td>0.675077</td>
      <td>0.367077</td>
      <td>1.080923</td>
      <td>0.681077</td>
      <td>0.849077</td>
      <td>...</td>
      <td>0.463077</td>
      <td>1.080923</td>
      <td>0.353077</td>
      <td>0.717077</td>
      <td>0.607077</td>
      <td>0.181077</td>
      <td>0.641077</td>
      <td>0.615077</td>
      <td>0.483077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.301077</td>
      <td>1.080923</td>
      <td>0.505077</td>
      <td>0.509077</td>
      <td>0.601077</td>
      <td>0.151077</td>
      <td>0.593077</td>
      <td>0.271077</td>
      <td>0.468923</td>
      <td>0.531077</td>
      <td>...</td>
      <td>0.623077</td>
      <td>0.073077</td>
      <td>0.096923</td>
      <td>0.922923</td>
      <td>0.194923</td>
      <td>0.382923</td>
      <td>0.489077</td>
      <td>0.407077</td>
      <td>0.775077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.479077</td>
      <td>1.080923</td>
      <td>0.354923</td>
      <td>0.260923</td>
      <td>0.281077</td>
      <td>0.299077</td>
      <td>0.243077</td>
      <td>0.585077</td>
      <td>0.861077</td>
      <td>0.789077</td>
      <td>...</td>
      <td>0.946923</td>
      <td>0.402923</td>
      <td>0.388923</td>
      <td>0.323077</td>
      <td>0.283077</td>
      <td>0.539077</td>
      <td>0.511077</td>
      <td>0.344923</td>
      <td>0.597077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.019077</td>
      <td>0.145077</td>
      <td>1.080923</td>
      <td>0.184923</td>
      <td>0.560923</td>
      <td>0.593077</td>
      <td>0.567077</td>
      <td>0.477077</td>
      <td>0.767077</td>
      <td>0.515077</td>
      <td>...</td>
      <td>0.156923</td>
      <td>0.622923</td>
      <td>0.052923</td>
      <td>1.080923</td>
      <td>0.528923</td>
      <td>0.188923</td>
      <td>0.143077</td>
      <td>0.391077</td>
      <td>0.275077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.572923</td>
      <td>0.124923</td>
      <td>0.132923</td>
      <td>0.421077</td>
      <td>0.617077</td>
      <td>0.385077</td>
      <td>0.753077</td>
      <td>0.833077</td>
      <td>0.699077</td>
      <td>0.583077</td>
      <td>...</td>
      <td>0.091077</td>
      <td>0.416923</td>
      <td>0.270923</td>
      <td>0.039077</td>
      <td>0.042923</td>
      <td>0.057077</td>
      <td>0.022923</td>
      <td>0.323077</td>
      <td>0.663077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.635077</td>
      <td>0.699077</td>
      <td>0.783077</td>
      <td>0.755077</td>
      <td>0.637077</td>
      <td>0.791077</td>
      <td>0.625077</td>
      <td>0.521077</td>
      <td>0.074923</td>
      <td>1.080923</td>
      <td>...</td>
      <td>0.041077</td>
      <td>0.000923</td>
      <td>0.013077</td>
      <td>0.430923</td>
      <td>0.580923</td>
      <td>0.732923</td>
      <td>0.192923</td>
      <td>0.138923</td>
      <td>0.308923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.028923</td>
      <td>0.227077</td>
      <td>0.609077</td>
      <td>0.607077</td>
      <td>0.403077</td>
      <td>0.407077</td>
      <td>0.262923</td>
      <td>0.616923</td>
      <td>1.080923</td>
      <td>0.982923</td>
      <td>...</td>
      <td>0.667077</td>
      <td>0.151077</td>
      <td>0.587077</td>
      <td>0.605077</td>
      <td>0.336923</td>
      <td>0.267077</td>
      <td>1.080923</td>
      <td>0.421077</td>
      <td>0.365077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.041077</td>
      <td>1.080923</td>
      <td>0.791077</td>
      <td>0.871077</td>
      <td>0.871077</td>
      <td>0.865077</td>
      <td>0.897077</td>
      <td>0.881077</td>
      <td>0.817077</td>
      <td>0.827077</td>
      <td>...</td>
      <td>0.837077</td>
      <td>0.843077</td>
      <td>0.533077</td>
      <td>1.080923</td>
      <td>0.104923</td>
      <td>0.735077</td>
      <td>0.807077</td>
      <td>0.827077</td>
      <td>0.781077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.853077</td>
      <td>0.899077</td>
      <td>0.893077</td>
      <td>0.821077</td>
      <td>0.877077</td>
      <td>0.743077</td>
      <td>0.881077</td>
      <td>0.429077</td>
      <td>1.080923</td>
      <td>0.715077</td>
      <td>...</td>
      <td>1.080923</td>
      <td>0.100923</td>
      <td>0.351077</td>
      <td>0.143077</td>
      <td>0.167077</td>
      <td>0.343077</td>
      <td>0.612923</td>
      <td>0.536923</td>
      <td>0.522923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.843077</td>
      <td>0.197077</td>
      <td>0.843077</td>
      <td>0.865077</td>
      <td>0.555077</td>
      <td>0.525077</td>
      <td>1.080923</td>
      <td>0.635077</td>
      <td>0.825077</td>
      <td>0.701077</td>
      <td>...</td>
      <td>0.144923</td>
      <td>1.080923</td>
      <td>0.635077</td>
      <td>0.753077</td>
      <td>0.837077</td>
      <td>0.795077</td>
      <td>0.887077</td>
      <td>0.845077</td>
      <td>0.579077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.946923</td>
      <td>0.986923</td>
      <td>0.349077</td>
      <td>0.395077</td>
      <td>0.051077</td>
      <td>0.011077</td>
      <td>0.055077</td>
      <td>0.095077</td>
      <td>0.152923</td>
      <td>0.236923</td>
      <td>...</td>
      <td>0.463077</td>
      <td>0.150923</td>
      <td>0.226923</td>
      <td>1.080923</td>
      <td>0.282923</td>
      <td>0.100923</td>
      <td>0.100923</td>
      <td>0.646923</td>
      <td>0.491077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.639077</td>
      <td>1.080923</td>
      <td>0.253077</td>
      <td>0.637077</td>
      <td>0.571077</td>
      <td>0.751077</td>
      <td>0.641077</td>
      <td>0.519077</td>
      <td>0.131077</td>
      <td>0.745077</td>
      <td>...</td>
      <td>0.433077</td>
      <td>0.269077</td>
      <td>0.517077</td>
      <td>0.098923</td>
      <td>0.209077</td>
      <td>0.417077</td>
      <td>0.483077</td>
      <td>0.393077</td>
      <td>1.080923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.521077</td>
      <td>1.080923</td>
      <td>0.481077</td>
      <td>0.829077</td>
      <td>0.901077</td>
      <td>0.855077</td>
      <td>0.883077</td>
      <td>0.863077</td>
      <td>0.677077</td>
      <td>0.861077</td>
      <td>...</td>
      <td>0.603077</td>
      <td>0.709077</td>
      <td>0.781077</td>
      <td>0.457077</td>
      <td>0.753077</td>
      <td>0.817077</td>
      <td>0.819077</td>
      <td>0.733077</td>
      <td>0.160923</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.530923</td>
      <td>1.080923</td>
      <td>0.411077</td>
      <td>0.152923</td>
      <td>0.389077</td>
      <td>0.583077</td>
      <td>0.705077</td>
      <td>0.531077</td>
      <td>0.771077</td>
      <td>0.779077</td>
      <td>...</td>
      <td>0.472923</td>
      <td>0.288923</td>
      <td>0.075077</td>
      <td>0.365077</td>
      <td>1.080923</td>
      <td>0.590923</td>
      <td>0.171077</td>
      <td>0.569077</td>
      <td>0.593077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.848923</td>
      <td>1.080923</td>
      <td>0.944923</td>
      <td>0.503077</td>
      <td>0.381077</td>
      <td>0.626923</td>
      <td>0.604923</td>
      <td>0.153077</td>
      <td>0.328923</td>
      <td>0.099077</td>
      <td>...</td>
      <td>0.775077</td>
      <td>0.789077</td>
      <td>1.080923</td>
      <td>0.741077</td>
      <td>0.793077</td>
      <td>0.729077</td>
      <td>0.627077</td>
      <td>0.376923</td>
      <td>0.823077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.819077</td>
      <td>0.747077</td>
      <td>1.080923</td>
      <td>0.501077</td>
      <td>0.061077</td>
      <td>0.551077</td>
      <td>0.087077</td>
      <td>0.719077</td>
      <td>0.835077</td>
      <td>0.585077</td>
      <td>...</td>
      <td>0.617077</td>
      <td>0.779077</td>
      <td>0.855077</td>
      <td>0.663077</td>
      <td>0.673077</td>
      <td>0.813077</td>
      <td>0.725077</td>
      <td>0.849077</td>
      <td>0.687077</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.067077</td>
      <td>0.605077</td>
      <td>0.715077</td>
      <td>0.681077</td>
      <td>0.611077</td>
      <td>1.080923</td>
      <td>0.503077</td>
      <td>0.227077</td>
      <td>0.691077</td>
      <td>0.653077</td>
      <td>...</td>
      <td>0.817077</td>
      <td>0.471077</td>
      <td>1.080923</td>
      <td>0.179077</td>
      <td>0.701077</td>
      <td>0.829077</td>
      <td>0.835077</td>
      <td>0.873077</td>
      <td>0.819077</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>139 rows × 217 columns</p>
</div>




```python
# Feed Euclidean Distances into SVC
feature_cols = [x for x in range(216)]

#grab all rows, and all cols
X = df_eucDist.loc[:, feature_cols]

# y response value in training set is always true, for 39 songs
y = np.ravel(np.array([[x] for x in df_eucDist['whoop']]))

#clf = SVC(kernel='linear', C=1.0)
#clf = SVC(kernel='rbf', C=1.0, gamma='auto')
clf = SVC(kernel='sigmoid', C=1.0)

xtrain = clf.fit(X,y)
test_size =100
clf.fit(X[:test_size],y[:test_size])
outcome = clf.predict(X[test_size:])
print(metrics.accuracy_score(y[test_size:], outcome))

print(y[test_size:])
print(outcome)
```

    0.6666666666666666
    [0 0 0 1 0 0 1 0 0 1 1 1 0 0 1 1 0 1 1 0 1 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0
     0 0]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0]
    

**Outcome 3:** Our SVC model on Euclidean distance has improved! However, the Euclidean Distance measure of similarity is susceptable to distortions in time. For this reason we will test another similarity measure that takes time shifts along an x axis into account - this will be known as Dynamic Time Warping, or DTW.


```python
import math
 
def dynamicTimeWarp(seqA, seqB, d = lambda x,y: abs(x-y)):
    # create the cost matrix
    numRows, numCols = len(seqA), len(seqB)
    cost = [[0 for _ in range(numCols)] for _ in range(numRows)]
 
    # initialize the first row and column
    cost[0][0] = d(seqA[0], seqB[0])
    for i in xrange(1, numRows):
        cost[i][0] = cost[i-1][0] + d(seqA[i], seqB[0])
 
    for j in xrange(1, numCols):
        cost[0][j] = cost[0][j-1] + d(seqA[0], seqB[j])
 
    # fill in the rest of the matrix
    for i in xrange(1, numRows):
        for j in xrange(1, numCols):
            choices = cost[i-1][j], cost[i][j-1], cost[i-1][j-1]
            cost[i][j] = min(choices) + d(seqA[i], seqB[j])
 
    for row in cost:
       for entry in row:
          print "%03d" % entry,
       print ""
    return cost[-1][-1]
```


      File "<ipython-input-24-acb3a314e55a>", line 24
        print "%03d" % entry,
                   ^
    SyntaxError: Missing parentheses in call to 'print'. Did you mean print("%03d" % entry, end=" ")?
    


**Outcome for Model 2:** Our model is now performing better, however it is not performing any better than a random prediction algorithm might. 


**Hypothesis 2:**  SVC here is predicting class values (y) based on an equal consolidation and coalition of pitches for each song, however we haven't given it a way to understand the underlying pitch patterns in a "whoop" segment as a measure of time and with respect to their note positions. 

**Question 3:** What if we derive further information from segments such as st. deviation, mean, coorelation, etc and use that as the input to our binary classification algorithms such as SVC? Would SVC perform better? If so how would SVC compare to like binary classification models?

**Question 4:** How would kNN perform with respect to our classification methods so far? 

**Question 5:** What is a good measure of the distance of the time patterns from eachother within whoop songs and from non-whoop songs? We might consider Euclidean distance, or Dynamic Time... 

**Question 6:** How might we enhance the data with additional time segment features such as timbre, and if we did how might this affect the accuracy of our model?

**Question 7:** How can we use this to identify MW occurances and frequency accross an entire song? And if we did, could we use MW occurance as a predictor of song popularity post Millenium? 

### RESULTS AND MODEL ACCURACY

* Here, given more time, we would summarize the predictive capabilities accross considered models, and we would use the top performing model on newly chosen song segments to see how it performs on entirely new data.

### DISCUSSION AND LIMITATIONS

Future versions and improvemens of this project may include:

* Compare SVC binary classifaction model to other methods of binary classification: Logistic Regression, Perceptron, NN, Linear Discrim. Analysis
* Utilize other audio analysis features to improve the model, such as the timbre pattern over MW segments, and see if a new combination of features improves the model performance.
* Measure "whoop" start time more precisely (currently being modelled using the same heuristic accross all songs: whoop start-1 second)
* Increase the training set and test set size by finding more songs that contain a MW
* Implement ROC curve to more accurately set binary classification threshold
* Improve data exploration with more involved graphing techniques, and in doing so graph elements of the Spotify API Audio Features JSON (in addition to the Audio Analysis JSON that was used here in project v1)
* May consider implementing the current F# code in python as to be included in this notebook
* Implement part B) of initially proposed project targets: determine whether the frequency of occurance of a MW can predict song popularity

Limitations and Considerations:

* The MW is constructed most often by a joining of the sounds "Ooo" -"Ah" -> In this project we have measured pitch only, cooresponding to the given frequency and therefore note patterns of a WM. Vowel sounds and other timbral measures have not been included in v1 of this project.  

### SUMMARY

In this project we learned how to collect, wrangle, visualize and model spotify audio analysis data. More specifically, we analysed pitches over time with respect to song segments that included common 3-5 chord progression pop phenomena known as the "Millenial Whoop." Unfortunately due to time constraints we have not been able to posulate a durable classification model which performs effectively. Further models will be investigated in later versions of this project. 

***
### ** ARCHIVE TO FOLLOW **
***

#### UNUSED CODE


```python
#plt.plot(df);
#data_dict = dict(zip((str(df['start'])), (df['pitches'])))
#data_dict = dict((str(pitch),pitch) for pitch in df['pitches'])
#py.plot(data_dict, filename='basic-scatter')
#scatter(df['pitches'], df['start']);

#df.plot(x='start', y='pitches')

#plt.scatter(df['start'], df['pitches'])
#plt.show()
#df.columns
#dataframe = pd.DataFrame(df['pitches'])
#dataframe

#df.T.plot( kind='bar') # or df.T.plot.bar()
#plt.show()
```


```python
file = 'C:\\Users\\Jnfr\\Projects\\SpotifyAnalyticsApp\\amy_winehouse_addicted.json'

with open(file, 'r') as json_data2:
    track = ijson.items(json_data2, 'track')
    trackCols = list(track)
    
df2 = pd.DataFrame.from_dict(json_normalize(trackCols))
```


```python
df2
```


```python
with open(file, 'r') as json_data3:
    sections = ijson.items(json_data3, 'sections')
    secCols = list(sections)
    
df3 = pd.DataFrame.from_dict(json_normalize(secCols))
```


```python
df3
```


```python
with open(file, 'r') as json_data3:
    beats = ijson.items(json_data3, 'beats')
    beatCols = list(beats)
    
df4 = pd.DataFrame.from_dict(json_normalize(beatCols))
```


```python
df4
```

#### CREATE FREQUENCY DATAFRAME


```python
data = {'octave_num': ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1','1','1','1','1','1','1','1','1','1',
                       '2', '2','2','2','2','2','2','2','2','2','2','2','3','3','3','3','3','3','3','3','3','3','3','3','4','4','4','4','4','4','4','4','4','4','4','4',
                       '5','5','5','5','5','5','5','5','5','5','5','5','6','6','6','6','6','6','6','6','6','6','6','6','7','7','7','7','7','7','7','7','7','7','7','7',
                       '8','8','8','8','8','8','8','8','8','8','8','8'], 'note': [ 'C', 'C#','D', 'Eb', 'E','F', 'F#', 'G', 'G#', 'A', 'Bb', 'B', 'C', 'C#','D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B',
                       'C', 'C#','D','Eb','E','F','F#', 'G', 'G#', 'A', 'Bb', 'B','C', 'C#','D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B',
                       'C', 'C#','D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B','C', 'C#','D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B',
                       'C', 'C#','D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B', 'C', 'C#','D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B',
                       'C', 'C#','D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B'],'frequency': [16.35,17.32,18.35,19.45,20.60,21.83,23.12,24.50,25.96,27.50,29.14,30.87,32.70,34.65,36.71,38.89,41.20,43.65,46.25,49.00,51.91,55.00,58.27,61.74,65.41,69.30,73.42,77.78,82.41,87.31,92.50,98.00,103.8,110.0,116.5,123.5,130.8,138.6,146.8,155.6,164.8,174.6,185.0,196.0,207.7,220.0,233.1,246.9,261.6,277.2,293.7,311.1,329.6,349.2,370.0,392.0,415.3,440.0,466.2,493.9,523.3,554.4,587.3,622.3,659.3,698.5,740.0,784.0,830.6,880.0,932.3,987.8,1047,1109,1175,1245,1319,1397,1480,1568,1661,1760,1865,1976,2093,2217,2349,2489,2637,2794,2960,3136,3322,3520,3729,3951,4186,4435,4699,4978,5274,5588,5920,6272,6645,7040,7459,7902]}


df_frequency = pd.DataFrame(data)
df_frequency.head()
```


```python
#load whoop data
import csv
from io import StringIO
import re
import pandas as pd

file_whoop = "c:\\Users\\Jnfr\\Projects\\SpotifyAnalyticsApp\\whoops.csv"

# get the data as a python string
with open (file_whoop, "r") as myfile:
    data=myfile.read()

# munge in python - get rid of the garbage in the input (lots of xff bytes)
data = re.sub(r'[^,a-zA-Z0-9_\.;:\n]', '', data) # get rid of the rubbish
#data = data + '\n' # the very last one is missing?
#data = re.sub(r';\n', r'\n', data) # last ; separator on line is problematic

# now let's suck into a pandas DataFrame
columns = ['artist', 'song_name', 'year', 'whoop_time', 'youtube_url']
df = pd.read_csv(StringIO(data), index_col=None, header=0, sep=',')
df.columns = columns
'''with open(file_whoop, 'r') as f:
    readCSV = csv.reader(f, delimiter=',')
    print(readCSV)
    #df = pd.DataFrame(object) '''


#df = pd.read_csv(file_whoop, header=0,encoding='iso-8859-1', error_bad_lines=False)
#df = pd.read_csv(StringIO(file_whoop), encoding = "cp1252")
df
#list = open(file_whoop, 'r')
#df_whoops = pd.DataFrame([list.read()])
#print(df_whoops)

```
