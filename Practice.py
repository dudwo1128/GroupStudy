f = open("C:/Users/dudwo/Documents/SideProject_CardUsage/Card_Usage_Update190225.csv", 'r', encoding='utf8')
lines = f.readlines()
for line in lines:
    line = line.strip()
    if line == ',,':
        pass
    else:
        print(line[1])
        #with open("test.csv",'a',encoding='utf-8') as w:
            #w.write(line)
f.close()