

# Dictionaries

def print_box(arg):
    print(arg)


print_box("Hello World!")
print_box("Enter a word:")
word = input()
if word == 'python':
    print_box("You entered Python!")
else:
    print_box("You didn't enter Python :(")


# Tutorial video file
# https://www.youtube.com/watch?v=yE9v9rt6ziw&ab_channel=ProgrammingwithMosh


import math

#Declairing Variable
student_count = 1000
student_name = "\tJon Smith"
rating = 4.99
is_published = True
course_name = "Python Programing"


#function call
print("\nHello World\n")
print(student_name)
print(student_name.upper())
print(student_name.lower())

print(math.ceil(2.2))


# simple function call 
def increment(number, by):
    return number + by

# function call with ** 
print(increment(234555,1324))
def save_user(**user):
    print(user)
save_user(id=1, name="John", age=22)
save_user(id=2, name="Smith", age=24)
