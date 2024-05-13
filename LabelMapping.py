# Produces types dictionary for the dataset

TYPESDICT = {}
# list all JPGs in folder:

if __name__ == "__main__":
    import os

    types = []

    files = os.listdir("FoodDatasetComplete")
    for file in files:
        if not file.endswith(".JPG"):
            continue
        splitName = file.split("_")
        if splitName[1] not in types:
            types.append(splitName[1])

    count = 0
    for type in types:
        TYPESDICT.update({type: count})
        count+=1
print(TYPESDICT)
