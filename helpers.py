# Funtion to load classnames from a text document
def load_class_names(file):
    class_names = []
    with open(file,'r') as f:
        lines = fp.readlines()

    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

