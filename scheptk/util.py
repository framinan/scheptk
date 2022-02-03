
import ast # to get the proper data type
import operator # for sorted functions
from random import randint # generation of random integer numbers

# utility function to get the proper type (int, string, float) of an argument
def get_proper_type(x):
    if isinstance(ast.literal_eval(x), int):
        return int(x)
    else:
        if isinstance(ast.literal_eval(x), float):
            return float(x)
        else:
            return x


# utility function to print the content of val (escalar, vector, matrix) with a tag
def print_tag(tag, value):
    # if it is a list
    if( isinstance(value, list)):
        # if it is a matrix
        if( isinstance(value[0],list)):
            print("[" + tag + "=" + matrix_to_string(value) + "]")
        else:
            # it is a vector (list)
            print("[" + tag + "=" + vector_to_string(value) + "]")
    else:
        # it is an scalar
        print("[" + tag + "=" + str(value) + "]")  


# utility function to map a vector into a string (mostly to be used internally)
def vector_to_string(vector):
    cadena = ''
    for i in range(len(vector)-1):
        cadena = cadena + str(vector[i]) + ","
    return cadena + str(vector[len(vector)-1]) 


# utility function to map a matrix into a string (mostly to be used internally)
def matrix_to_string(matrix):
    cadena = ''
    for i in range(len(matrix)-1):
        for j in range(len(matrix[i]) - 1):
            cadena = cadena + str(matrix[i][j]) + ","
        cadena = cadena + str(matrix[i][len(matrix[i])-1]) + ";"
    # last row
    for j in range(len(matrix[0])-1):
        cadena = cadena + str(matrix[len(matrix)-1][j]) + ","
    # last col of last row
    return cadena + str(matrix[len(matrix)-1][len(matrix[0])-1]) 
   

# utility function to read a tag
def read_tag(filename, tag):
    with open(filename) as file:
        lines = file.readlines()
        line_number = -1
        found = -1
        while(line_number < len(lines)-1 and found == -1):
            line_number = line_number+1
            found = lines[line_number].find("["+tag + "=")
        if(found ==-1):
            print("Tag " + tag + " does not exist in file " + filename + ". ", end="")
            return -1
        else:
            # The tag exists, now it is loaded
            tag_value = lines[line_number][found + len(tag)+2:lines[line_number].find("]")]
            # create the proper data structure
            # scalar
            if(tag_value.find(",") ==-1):
                return int(tag_value)
            else:
                if(tag_value.find(";") !=-1):
                    values = []
                    rows = tag_value.split(';')
                    for i in range(len(rows)):
                        content = rows[i].split(',')
                        values.append([get_proper_type(e) for e in content])                        
                    return values
                else:
                    content = tag_value.split(',')
                    values = [get_proper_type(e) for e in content]
                    return values


# utility function to obtain the index of a sorted list. It is used internally, most of the times use sorted_index_asc or sorted_index_desc
def sorted_index(list, descending):
    tuple = enumerate(list)
    sorted_tuple = sorted(tuple, key=operator.itemgetter(1), reverse=descending)
    sorted_index = [index for index,item in sorted_tuple]
    return sorted_index

def sorted_index_asc(list):
    return sorted_index(list, False)

def sorted_index_desc(list):
    return sorted_index(list, True)


# utility function to obtain the value of a sorted list. It is used internally, most of the times use sorted_value_asc or sorted_value_desc
def sorted_value(list, descending):
    tuple = enumerate(list)
    sorted_tuple = sorted(tuple, key=operator.itemgetter(1), reverse=descending)
    sorted_list = [item for index,item in sorted_tuple]
    return sorted_list    

def sorted_value_asc(list):
    return sorted_value(list, False)

def sorted_value_desc(list):
    return sorted_value(list, True)

# utility function to generate a random sequence of length size
def random_sequence(size):
    sequence = []
    for i in range(size):
        number = randint(0, size-1)
        while( sequence.count(number)!= 0):
            number = randint(0, size-1)
        sequence.append(number)
    return sequence


# find_index_max() returns the index where the maximum value is found
def find_index_max(list):
    tuple = enumerate(list)
    sorted_tuple = sorted(tuple, key=operator.itemgetter(1), reverse=True)
    return sorted_tuple[0][0]

# find_index_min() returns the index where the minimum value is found
def find_index_min(list):
    tuple = enumerate(list)
    sorted_tuple = sorted(tuple, key=operator.itemgetter(1), reverse=False)
    return sorted_tuple[0][0]

# utility function to write a tag (scalar, vector, matrix) and its value into a file
def write_tag(tag, value, filename):
    with open(filename, 'a') as file:
        file.write('[' + tag + '=')

        if(isinstance(value, list) ):
            # if it is a list, let's see if it is a vector or a matrix
            if(isinstance(value[0], list) ):
                # matrix:
                file.write(matrix_to_string(value) + "]\n")
            else:
                # vector
                file.write(vector_to_string(value) + "]\n")
        else:
            # value
            file.write(str(value) + "]\n")    