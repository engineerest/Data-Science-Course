numbers = [1, 13, 4, 5, 63, 100]
print(numbers)

new_numbers = []

for n in numbers:
    new_numbers.append(n*2)

print(new_numbers)

new_numbers = [n*2 for n in numbers]
print(new_numbers)

for i in range(2):
    for j in range(5):
        print(i+j, end=" ")


new_list_comprehension_1 = [i+j for i in range(2) for j in range(5)]
print(new_list_comprehension_1)

print(type(new_list_comprehension_1))
print(new_list_comprehension_1)

new_list_comprehension_2 = [i+j for i in range(2) for j in range(5)]
print(new_list_comprehension_2)

print(type(new_list_comprehension_2))

print(type(new_list_comprehension_1[1]))

print(list(range(1, 11)))

print([num ** 3 for num in range(1, 11) if num % 2 != 0])

for i in range(1, 11):
    if i % 2 != 0:
        print(i**3, end=" ")

# print([num ** 3 if num % 2 != 0 for num in range(1, 11)]) error
print([num ** 3 if num % 2 != 0 else "even" for num in range(1, 11)])

# List Comprehensions
# - a very powerful tool (it can be applied to a very wide) range of cases and can deliver many types of output
# - a fantastic example of high-quality code
# - require more memory and run more slowly

