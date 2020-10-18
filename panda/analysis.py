import pandas as pd

poke = pd.read_csv('pokemon_data.csv')

print(poke.head(10))
print(poke.tail(50))
