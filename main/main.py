# Test Function script

def my_function(fname):
  print(fname + " Refsnes")

my_function("Emil")
my_function("Tobias")
my_function("Linus")


def my_function(*kids):
  print("The youngest child is " + kids[2])

my_function("Emil", "Tobias", "Linus")
# function call 


def my_function(x):
  return 5 * x

print(my_function(3))
print(my_function(5))
print(my_function(9))



