import src.PCA as PCA
from PIL import Image

path = "E:/Study/Project/Python/xla/PCA_Face/images/1/0.png"
image = Image.open(path)
image.show()
out = PCA.PCA(path)
print(out)
# show 5 image ket qua
for i in range(len(out)):
    im = Image.open(out[i])
    im.show()
