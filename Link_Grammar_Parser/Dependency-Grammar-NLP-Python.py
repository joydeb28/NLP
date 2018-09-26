# Dependency-Grammar-Parser-NLP-Python



# Dependency-Grammar-NLP-Python
from pylinkgrammar.linkgrammar import Parser
p = Parser()
s2 = "Most current machine learning works well because of human-designed representations and input features."
def print_link(s):
    s= str(s)
    s1 = s.split(": ")
    s = s1[1]
    w = s.split('-')
    if(len(w)>3 and w[0]=='LEFT'):
        print("%s\t%s\t%s\t\t%s\t\t"%(w[0]+"-"+w[1],"-",w[2],w[3]))
    elif(len(w)>3 and w[0]!='LEFT'):
        print("%s\t\t%s\t%s\t\t%s\t"%(w[0],"-",w[1]+"-"+w[2],w[3]))   
    else:
        print("%s\t\t%s\t%s\t%s\t%s"%(w[0],"-",w[1],"-",w[2]))

linkages = p.parse_sent(s2)
print linkages[0].diagram
links_all =  linkages[0].links
#print links_all
for i in links_all:
    print_link(i)
