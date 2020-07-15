# Condition logic

price = 100000

weight = int(input('Weight: '))
unit = input('(L)bs or (K)g: ')


# Logical opperative
has_high_income = True
has_good_cc = True

if has_high_income and not has_good_cc:
    print("Yes please give him the loan !")

if has_high_income or has_good_cc:
    print("No please give him the loan !")

if has_good_cc:
    down_pay = 0.341 * price
else:
    down_pay = 0.56782 * price

print(f"Down Payment: ${down_pay}")

if unit.upper() == "L":
    converted = weight * 0.45
    print(converted)
else:
    converted = weight / 0.45
    print(converted)

k = 1
while k <= 5:
    print("Converting...")
    k = k + 1
print("Done!")


secret_number = 9
guess_count = 0
guess_limit = 3

while guess_count < guess_limit:
    guess = int(input('Guess: '))
    guess_count += 1
    if guess == secret_number:
        print('You Won')
    else:
        print('Try again')
