import sys

from scipy import misc as scpm
from kmeans import kmeans
 

def main():
    if len(sys.argv) < 2:
        return
    img = scpm.imread(sys.argv[1])
    
    print(img.shape)
    n, m, dim = img.shape
    img = img.reshape(-1, dim)

    C, a = kmeans(img, 10)

    img = C[a]
    img = img.reshape(n, m, dim)
    if len(sys.argv) > 2:
        scpm.imsave(sys.argv[2], img)
    else:
        scpm.imshow(img)




    

if __name__ == '__main__':
    main()



