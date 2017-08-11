# KBSNNPy
Running the Web App:

`1. Setup Environment`
```bash
$ git clone https://github.com/kanarelo/KBSNNPy.git
$ cd KBSNNPy/kbs_web

$ virtualenv ~/.venvs/KBSNNPy
$ source ~/.venvs/KBSNNPy/bin/activate

$ pip install -r requirements.txt
```

`2. Running in Ipython:`
```python
>>> from kbs_web.nn import KBSPurchaseRegressor

>>> regressor = KBSPurchaseRegressor()
>>> regressor.setup_model()
>>> regressor.train_model()

>>> regressor.predict()
[3, 5, 6]
```

`3. Running in commandline`
```bash
$ python main.py
```

`3. Running the web version`
```bash
$ python manage.py runserver
Performing system checks...

Using TensorFlow backend.
System check identified no issues (0 silenced).
August 11, 2017 - 05:05:51
Django version 1.10.6, using settings 'kbs_web.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```
