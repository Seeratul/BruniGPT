#A convenient File for arbitrary code execution tangentially related to the project
#Unthrifty loveliness why dost thou spend, Upon thy self thy beauty's legacy?
f = open("sd_train.txt")
text = f.read()
f.close()
f = open("shakes.txt")
otext = f.read()
f.close()
print(len(text))
print(len(otext))