import random

def calculdateBMI(height, weight):
    BMI = weight / ((height / 100) ** 2)
    if BMI < 18.5 :
        return 'thin'
    elif BMI < 25 :
        return 'normal'
    else :
        return 'fat'

with open('data\\bmi.csv', 'w', encoding='UTF-8') as file1:
    file1.write('height,weight,label\n')
    category = {'thin': 0, 'normal': 0, 'fat': 0}
    for i in range(30000):
        height = random.randint(120, 200)
        weight = random.randint(40, 70)
        label = calculdateBMI(height, weight)
        category[label] += 1
        file1.write("%d,%d,%s\n" % (height, weight, label))

    print(f"generate OK. result={category}")