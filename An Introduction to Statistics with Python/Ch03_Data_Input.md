
# 3 Data Input
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [3 Data Input](#3-data-input)
  * [3.1 Input from Text Files](#31-input-from-text-files)
    * [3.1.1 Visual Inspection](#311-visual-inspection)
    * [3.1.2 Reading ASCII-Data into Python](#312-reading-ascii-data-into-python)
  * [3.2 Input from MS Excel](#32-input-from-ms-excel)
  * [3.3 Input from Other Formats](#33-input-from-other-formats)
    * [3.3.1 Matlab](#331-matlab)

<!-- tocstop -->


## 3.1 Input from Text Files

### 3.1.1 Visual Inspection

### 3.1.2 Reading ASCII-Data into Python


```python
import sys, os
import pandas as pd
# cd 'C:\Data\storage'
sys.path.append("./Data/storage/")
```


```python
pwd # Check if you were successful
```




    'I:\\BOOKS\\SC_STATS\\PYTHON\\An Introduction to Statistics with Python\\SF_ISP'




```python
#ls # List the files in that directory
inFile = 'data.txt'
```




    'data.txt'




```python
df = pd.read_csv('Data/storage/'+inFile)
```


```python
df.head() # Check if first line is ok
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>1.3</th>
      <th>0.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2.1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>4.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>3.3</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail() # Check the last line
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>1.3</th>
      <th>0.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2.1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>4.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>3.3</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>



* a) Simple Text-Files


```python
import numpy as np
data = np.loadtxt('Data/storage/data.txt', delimiter=',')
data
```




    array([[ 1. ,  1.3,  0.6],
           [ 2. ,  2.1,  0.7],
           [ 3. ,  4.8,  0.8],
           [ 4. ,  3.3,  0.9]])




```python
df = pd.read_csv('Data/storage/data.txt', header=None)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.3</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2.1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3.3</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv('Data/storage/data.txt')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>1.3</th>
      <th>0.6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2.1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>4.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>3.3</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

* b) More Complex Text-Files


```python
df2 = pd.read_csv('Data/storage/data2.txt', skipfooter=1, delimiter='[ ,] *')
df2
```

    C:\Anaconda36\lib\site-packages\ipykernel\__main__.py:1: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support skipfooter; you can avoid this warning by specifying engine='python'.
      if __name__ == '__main__':





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Weight</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.3</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2.1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.8</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>



* c) Regular Expressions

Working with text data often requires the use of simple regular expressions.
Regular expressions are a very powerful way of finding and/or manipulating text strings.
Many books have been written about them, and good, concise information on regular expressions can be found on the web, for example at:
* https://www.debuggex.com/cheatsheet/regex/python provides a convenient cheat sheet for regular expressions in Python.
* http://www.regular-expressions.info gives a comprehensive description of regular expressions.

Let me give two examples how pandas can make use of regular expressions:
1. Reading in data from a file, separated by a combination of commas, semicolons, or white-spaces:
```
df = pd.read_csv(inFile, sep='[ ;,]+')
```
The square brackets indicate a combination ```(“[:: :]”)``` of ```:::```
The plus indicates one or more (“+”)
2. Extracting columns with certain name-patterns from a pandas DataFrame. In the following example, I will extract all the columns starting with Vel:



```python
data = np.round(np.random.randn(100,7), 2)
```


```python
df = pd.DataFrame(data, columns=['Time','PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ'])
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>PosX</th>
      <th>PosY</th>
      <th>PosZ</th>
      <th>VelX</th>
      <th>VelY</th>
      <th>VelZ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.40</td>
      <td>0.50</td>
      <td>-0.05</td>
      <td>3.18</td>
      <td>-1.35</td>
      <td>1.01</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.50</td>
      <td>1.70</td>
      <td>-0.28</td>
      <td>-1.26</td>
      <td>-0.00</td>
      <td>0.53</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.04</td>
      <td>1.01</td>
      <td>0.56</td>
      <td>-1.07</td>
      <td>-1.42</td>
      <td>-0.28</td>
      <td>-1.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.96</td>
      <td>-2.86</td>
      <td>1.32</td>
      <td>-0.03</td>
      <td>-1.02</td>
      <td>-0.10</td>
      <td>-0.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.84</td>
      <td>1.25</td>
      <td>-1.71</td>
      <td>-0.04</td>
      <td>1.49</td>
      <td>-0.71</td>
      <td>-0.59</td>
    </tr>
  </tbody>
</table>
</div>




```python
vel = df.filter(regex='Vel*')
vel.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VelX</th>
      <th>VelY</th>
      <th>VelZ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.35</td>
      <td>1.01</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.00</td>
      <td>0.53</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.42</td>
      <td>-0.28</td>
      <td>-1.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.02</td>
      <td>-0.10</td>
      <td>-0.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.49</td>
      <td>-0.71</td>
      <td>-0.59</td>
    </tr>
  </tbody>
</table>
</div>



## 3.2 Input from MS Excel

There are two approaches to reading a Microsoft Excel file in pandas: the function
read_excel, and the class ExcelFile.1
* read_excel is for reading one file with file-specific arguments (i.e., identical data formats across sheets).
* ExcelFile is for reading one file with sheet-specific arguments (i.e., different data formats across sheets).
Choosing the approach is largely a question of code readability and execution speed.

The following commands show equivalent class and function approaches to read a single sheet:
```py
# using the ExcelFile class
xls = pd.ExcelFile('path_to_file.xls')
data = xls.parse('Sheet1', index_col=None,
na_values=['NA'])

# using the read_excel function
data = pd.read_excel('path_to_file.xls', 'Sheet1', index_col=None, na_values=['NA'])
```

If this fails, give it a try with the Python package ```xlrd```.
The following advanced script shows how to directly import data from an Excel file which is stored in a zipped archive on the web:



```python
# %load ch03/L3_2_readZip.py
'''Get data from MS-Excel files, which are stored zipped on the WWW. '''

# author: Thomas Haslwanter, date: Nov-2015

# Import standard packages
import pandas as pd

# additional packages
import io
import zipfile

# Python 2/3 use different packages for "urlopen"
import sys
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

def getDataDobson(url, inFile):
    '''Extract data from a zipped-archive on the web'''

    # get the zip-archive
    GLM_archive = urlopen(url).read()

    # make the archive available as a byte-stream
    zipdata = io.BytesIO()
    zipdata.write(GLM_archive)

    # extract the requested file from the archive, as a pandas XLS-file
    myzipfile = zipfile.ZipFile(zipdata)
    xlsfile = myzipfile.open(inFile)

    # read the xls-file into Python, using Pandas, and return the extracted data
    xls = pd.ExcelFile(xlsfile)
    df  = xls.parse('Sheet1', skiprows=2)

    return df

if __name__ == '__main__':
    # Select archive (on the web) and the file in the archive
    url = 'http://cdn.crcpress.com/downloads/C9500/GLM_data.zip'
    inFile = r'GLM_data/Table 2.8 Waist loss.xls'

    df = getDataDobson(url, inFile)
    print(df)

    input('All done!')

```

        man  before  after
    0     1   100.8   97.0
    1     2   102.0  107.5
    2     3   105.9   97.0
    3     4   108.0  108.0
    4     5    92.0   84.0
    5     6   116.7  111.5
    6     7   110.2  102.5
    7     8   135.0  127.5
    8     9   123.5  118.5
    9    10    95.0   94.2
    10   11   105.0  105.0
    11   12    85.0   82.4
    12   13   107.2   98.2
    13   14    80.0   83.6
    14   15   115.1  115.0
    15   16   103.5  103.0
    16   17    82.0   80.0
    17   18   101.5  101.5
    18   19   103.5  102.6
    19   20    93.0   93.0
    All done!1


## 3.3 Input from Other Formats

* __Matlab__  Support for data input from Matlab files is built into scipy, with the command __scipy.io.loadmat__.
* __Clipboard__ If you have data in your clipboard, you can import them directly with __pd.read_clipboard()__
* __Otherfile formats__ Also SQL databases and a number of additional formats are supported by pandas.
The simplest way to access them is typing __pd.read_ + TAB__, which shows all currently available options for reading data into pandas DataFrames.


### 3.3.1 Matlab

The following commands return string, number, vector, and matrix variables from a Matlab file “data.mat”, as well as the content of a structure with two entries (a vector and a string).
The Matlab variables containing the scalar, string, vector, matrix, and structure are called number, text, vector, matrix, and structure, respectively.




```python
from scipy.io import loadmat
data = loadmat('Data/storage/data.mat')
```


```python
data
```




    {'__globals__': [],
     '__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Sun Apr 02 23:18:49 2017',
     '__version__': '1.0',
     'data': array([[ 1. ,  1.3,  0.6],
            [ 2. ,  2.1,  0.7],
            [ 3. ,  4.8,  0.8],
            [ 4. ,  3.3,  0.9]])}




```python
number = data['number'][0,0]
text = data['text'][0]
vector = data['vector'][0]
matrix = data['matrix']
struct_values = data['structure'][0,0][0][0]
strunct_string = data['structure'][0,0][1][0]
```


```python

```
