import re
import csv
import itertools
import string

"""
https://keras.io/getting-started/sequential-model-guide/
https://keras.io/optimizers/
https://www.quora.com/In-Keras-what-is-a-dense-and-a-dropout-layer
http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
https://keras.io/getting-started/sequential-model-guide/
https://github.com/miloharper/multi-layer-neural-network/blob/master/main.py
"""

def get_sales_by_year():
    rows = sorted(list((" ".join(r[:2]), r[2]) for r in csv.reader(open("fleet2.csv")))[1:])
    dataset = [(rows[i][0], int(rows[i][1]), calculate_number_of_plates(rows[i][0], rows[i + 1][0])) for i in range(len(rows) - 1)]

    sales_by_year = {}
    for plate, year, number_of_cars in dataset:
        previous_values = sales_by_year.get(year)

        if previous_values is not None:
            next_values = (previous_values[0] + number_of_cars, previous_values[1] + 1)
        else:
            next_values = (number_of_cars, 1)

        sales_by_year[year] = next_values

    return sales_by_year

def clean_data():
    data =  get_sales_by_year()
    items = sorted(data.items(), key=lambda x:x[0])
    for year, (total_purchases, frequency) in items:
        yield year, total_purchases, frequency

def extract_number_plate(sentence):
    """
    Take in a string and extract a
    Kenyan vehicle number plate.
    >>> extract_number_plate("KBL 468B")
    ["KBL 468B"]
    >>> extract_number_plate("GBS 333")
    []
    >>> extract_number_plate("KRE 635")
    ["KRE 635"]
    >>> extract_number_plate("KTB 222")
    @return a list of number plates
    """
    number_plate_regex_pattern = r"(K[A-Z]{2}\ [0-9]{3}[A-Z]{0,1})"
    return re.findall(number_plate_regex_pattern, sentence.upper())

_cache = None
def calculate_number_of_plates(plate_a, plate_b):
     global _cache

     def generate_plate_numbers():
         """
         Generate the test values
         """
         for a in string.ascii_uppercase:
             for b in string.ascii_uppercase:
                 for i in range(10):
                     for j in range(10):
                         for k in range(10):
                             for c in string.ascii_uppercase:
                                 yield "K%s%s %d%d%d%s" % (a, b, i, j, k, c)

     if _cache is None:
         _cache = list(generate_plate_numbers())
     plates = _cache

     return abs(plates.index(plate_a) - plates.index(plate_b))
