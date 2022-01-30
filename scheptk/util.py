import ast # to get the proper data type



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
            print("Tag " + tag + " does not exist in file " + filename)
            return -1
        else:
            tag_value = lines[line_number][found + len(tag)+2:len(lines[line_number])-2]
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


# utility function to get the proper type (int, string, float) of an argument
def get_proper_type(x):
    if isinstance(ast.literal_eval(x), int):
        return int(x)
    else:
        if isinstance(ast.literal_eval(x), float):
            return float(x)
        else:
            return x