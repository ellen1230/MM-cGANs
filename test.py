

f = open('name.txt')
names = []
genders = []

for s in f.readlines():
    s=s.strip('\n')
    names.append(s.split('-')[0])
    gender = s.split('-')[-1]
    if gender == 'women':
        genders.append(str(0))
    elif gender == 'men':
        genders.append(str(1))

f.close()
