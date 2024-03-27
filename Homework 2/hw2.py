# Timothy Horrell
# ECE 1770
# Homework 2

# Imports
import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt

#--------------------------------
#           Problem 1
#--------------------------------

# problem 1a
range_array = np.arange(100,201,20,dtype=int)
range_array_1 = np.copy(range_array)
range_array = np.transpose(np.vstack((range_array, range_array_1)))
print(f'\n6x2 Integer array: ',range_array)

# problem 1b
ev_odd_array = np.arange(3,61,3,dtype=int)
ev_odd_array = ev_odd_array.reshape((5,4))
ev_odd_array = ev_odd_array[1::2, ::2]
print(f'\nEven Odd Array: ',ev_odd_array)

#--------------------------------
#           Problem 2
#--------------------------------

# problem 2a
point_A = np.array(([1,3], [2,5]))

euclidean_distance = distance.pdist(point_A, 'euclidean')
cosine_distance = distance.pdist(point_A, 'cosine')
hamming_distance = distance.pdist(point_A, 'hamming')
print(f'\nEuclidean Distance:',euclidean_distance,'\nCosine Distance:',cosine_distance,'\nHamming Distance:',hamming_distance)

# problem 2b
elim_array = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
elim_csr = csr_matrix(elim_array)
print(f'\nCSR Matrix:',elim_csr)
print(f'\nNon-zero count:',csr_matrix.count_nonzero(elim_csr))
elim_csr.eliminate_zeros()
print(f'\nZero-less CSR Matrix:',elim_csr,'\n\n')

#--------------------------------
#           Problem 3
#--------------------------------

company_sales = pd.read_csv('input/company_sales_data.csv')
print(company_sales.head())

#plotting part
company_sales.plot(x='month_number', y=['facecream','facewash','toothpaste','bathingsoap','shampoo','moisturizer'], figsize=(10,5))
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Company Sales Data')
plt.savefig('output/hw2_3a')

#stack plot
company_sales.plot.area(x='month_number', y=['facecream','facewash','toothpaste','bathingsoap','shampoo','moisturizer'], figsize=(10,5))
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Total Company Sales Data')
plt.savefig('output/hw2_3b')

#--------------------------------
#           Problem 4
#--------------------------------

#bar chart
company_sales.plot.bar(x='month_number', y=['facecream','facewash'], figsize=(10,5))
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Facecream and Facewash Monthly Sales Data')
plt.savefig('output/hw2_4a')

#create subplot
fig, axes = plt.subplots(nrows=2, ncols=1)
company_sales.plot.bar(x='month_number', y=['facecream','facewash'], figsize=(10,5), ax=axes[0])
company_sales.plot.bar(x='month_number', y=['facecream','facewash'], figsize=(10,5), ax=axes[1])
plt.xlabel('Month')
plt.ylabel('Sales')
plt.savefig('output/hw2_4b')

#--------------------------------
#           Problem 5
#--------------------------------

#pie plot
plt.clf()
company_sales_totals = company_sales.drop(columns=['month_number', 'total_units', 'total_profit']).sum()
company_sales_totals.plot.pie(figsize=(10,5))
plt.title('Yearly Sales per Product')
plt.savefig('output/hw2_5a')

#histogram
company_sales.plot.hist(column='total_profit')
plt.title('Total Profit Histogram')
plt.savefig('output/hw2_5b')