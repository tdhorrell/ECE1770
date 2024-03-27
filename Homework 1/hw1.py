#Tim Horrell
#ECE 1770 HW1

# Run this code in terminal using Code Runner to accept user input

#--------------------------------
#           Problem 1
#--------------------------------

#1a
print("This program will dipslay the sum of the current and previous number up to value N.")
input_number = int(input("Value for N: "))
print([2*val - 1 for val in range(input_number)])

#1b
print("\nThis program will dipslay characters at even indicies of the given string.\n**This assumes 0 is an even index.")
input_string = input("Test string: ")
print([val for i, val in enumerate(input_string) if (i%2 == 0)])

#--------------------------------
#           Problem 2
#--------------------------------

print("\nThis program will check if the first and last numbers are equal and then displays the numbers divisible by 5.")

input_list = []
n = int(input("Enter the number of elements in the list: "))
for i in range(0,n):
    input_list.append(int(input("Enter one element: ")))

#define function
def first_last_div(input_list):
    print([val for val in input_list if (val%5==0)])
    return input_list[0] == input_list[-1]

print(first_last_div(input_list))

#--------------------------------
#           Problem 3
#--------------------------------

print("\nThis program will check if the given number is a palindrome.")
input_palindrome = int(input("Input palindrome: "))
print(str(input_palindrome) == str(input_palindrome)[::-1])

#--------------------------------
#           Problem 4
#--------------------------------

print("\nThis program will combine the even numbers of one list with the odd numbers of the other.")

input_list_1 = []
input_list_2 = []

n = int(input("Enter the number of elements in the odd list: "))
for i in range(0,n):
    input_list_1.append(int(input("Enter one element: ")))

n = int(input("Enter the number of elements in the even list: "))
for i in range(0,n):
    input_list_2.append(int(input("Enter one element: ")))

print([val for val in input_list_1 if val%2==1] + [val for val in input_list_2 if val%2==0])

#--------------------------------
#           Problem 5
#--------------------------------

print("\nThis program will print the values of an integer backwards separated by a space.")
input_int = input("Input int: ")
print(' '.join([val for val in str(input_int)[::-1]]))