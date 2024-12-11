import csv
import warnings
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

# Ouvrir le fichier de données
# Open the data file 打开数据文件
z = []
nom = []
with open('/Users/hanyitian/Desktop/ONIP_Bloc2_Laser/data.csv','r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        z = np.append(z,float(row[0]))
        nom = np.append(nom,row[1])

# Lire les images
# Read the images 读取图像
image = []
image.append(16)
image[0] = plt.imread(nom[0])
print("For image 1 :")
plt.imshow(image[0])

# Calculer le centre de gravité de l'image
# Calculate the center of gravity of the image 计算图像重心
image_array = np.array(image[0])
X = np.linspace(0,499,500)
Y = np.linspace(0,499,500)
XX,YY = np.meshgrid(X,Y)
I_total = np.sum(image_array)
x_bary = np.sum(XX * image_array) / I_total
y_bary = np.sum(YY * image_array) / I_total
print("X_bary = ",x_bary," Y_bary = ",y_bary)

# Calculer les valeurs maximale et minimale de l'image
# Calculate the maximum and minimum values of the image 计算图像的最大最小值
I_max = np.max(image_array)
I_min = np.min(image_array)
print("I_max = ",I_max," I_min = ",I_min)

# Vérifier le point d'intersection
# Verify the intersection point 验证交点
plt.axvline(x=x_bary,color='g')
plt.axhline(y=y_bary,color='g')
plt.scatter(x=x_bary,y=y_bary,color='r')
plt.show()

# Tracer le profil
# Plot the profile 绘制剖面图
x_bary_int = int(x_bary)
y_bary_int = int(y_bary)
vertical = image_array[:, x_bary_int]
horizontal = image_array[y_bary_int, :]
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(vertical, color='b')
plt.title('Vertical (x = x_bary)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Row Index (y)')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(horizontal, color='r')
plt.title('Horizontal (y = y_bary)')
plt.xlabel('Column Index (x)')
plt.ylabel('Pixel Intensity')
plt.grid()
plt.tight_layout()
plt.show()

# Définir le fonction gaussienne
# Define the Gaussian function 定义高斯函数
def gaussian(x, A, B, x0, omega):
    return A + B * np.exp(-2 * ((x - x0) ** 2) / (omega ** 2))

# Ajustement du profil longitudinal
# Fit the vertical profile 纵向剖面拟合
x_vertical = np.linspace(0,499,500)
p0_vertical = [I_min, I_max-I_min, y_bary_int, 120]
params_vertical,_ = curve_fit(gaussian, x_vertical, vertical, p0_vertical)
print("Vertical fit parameters (A, B, x0, omega) :", params_vertical)

# Ajustment du profil transversal
# Fit the horizontal profile 横向剖面拟合
x_horizontal = np.linspace(0,499,500)
p0_horizontal = [I_min, I_max-I_min, x_bary_int, 120]
params_horizontal,_ = curve_fit(gaussian, x_horizontal, horizontal, p0_horizontal)
print("Horizontal fit parameters (A, B, x0, omega) :", params_horizontal)

# Tracer le profil ajusté
# Plot the fitted profile 绘制拟合剖面图
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x_vertical, gaussian(x_vertical, *params_vertical), 'b-', label='Gaussian Fit')
plt.title('Vertical Gaussian Fit')
plt.xlabel('Row Index (y)')
plt.ylabel('Pixel Intensity')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(x_horizontal, gaussian(x_horizontal, *params_horizontal), 'r-', label='Gaussian Fit')
plt.title('Horizontal Gaussian Fit')
plt.xlabel('Column Index (x)')
plt.ylabel('Pixel Intensity')
plt.grid()
plt.tight_layout()
plt.show()

# Calculer la taille réelle du faisceau lumineux
# Calculate the actual size of the beam 计算光束实际尺寸
x_size = params_horizontal[3] * 4.65
y_size = params_vertical[3] * 4.65
print("x_size = ",x_size, "um")
print("y_size = ",y_size, "um")

# Traiter les images restantes
# Process the remaining 15 images 对剩余15张图片进行处理
omega_x = []
omega_y = []
for i in range(15):
    img = plt.imread(nom[i])
    image.append(img)
    image_array = np.array(img)
    x = np.linspace(0,499,500)
    y = np.linspace(0,499,500)
    XX,YY = np.meshgrid(x,y)
    I_total = np.sum(image_array)
    x_bary = np.sum(XX * image_array) / I_total
    y_bary = np.sum(YY * image_array) / I_total
    x_bary_int = int(x_bary)
    y_bary_int = int(y_bary)
    vertical = image_array[:, x_bary_int]
    horizontal = image_array[y_bary_int, :]
    I_max = np.max(image_array)
    I_min = np.min(image_array)
    p0_vertical = [I_min, I_max - I_min, y_bary_int, 120]
    p0_horizontal = [I_min, I_max - I_min, x_bary_int, 120]
    params_vertical, _ = curve_fit(gaussian, x, vertical, p0_vertical)
    params_horizontal, _ = curve_fit(gaussian, x, horizontal, p0_horizontal)
    omega_x.append(params_horizontal[3])
    omega_y.append(params_vertical[3])
    omega_x = [float(x) for x in omega_x]
    omega_y = [float(y) for y in omega_y]

# Créer une fonction rayon
# Create a rayon function 创建rayon函数
wavelength = 1.3 * 1e-6
def rayon(z,omega_0,M2,wavelength):
    return omega_0 * np.sqrt(1 + (z * M2 * wavelength / (np.pi * omega_0**2))**2)

# Ajuster omega_0 et M^2 en utilisant la méthode des moindres carrés
# Fit omega_0 and M^2 using the least squares method 利用最小二乘法拟合omega_0和M^2
p0_initial = [70,1.5]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=OptimizeWarning)
    # On ignore ici l'avertissement dû au calcul de la covariance, car en réalité nous n'en avons pas besoins
    params_x,_ = curve_fit(lambda z,omega_0_x,M2_x:rayon(z,omega_0_x,M2_x,wavelength),z,omega_x,p0_initial)
    params_y,_ = curve_fit(lambda z,omega_0_y,M2_y:rayon(z,omega_0_y,M2_y,wavelength),z,omega_y,p0_initial)
omega_0_x_fit,M2_x_fit = params_x
omega_0_y_fit,M2_y_fit = params_y
print("For all images :")
print("omega_0 fit in label x = ",omega_0_x_fit,"um,  M^2 fit in label x = ",M2_x_fit)
print("omega_0 fit in label y = ",omega_0_y_fit,"um,  M^2 fit in label y = ",M2_y_fit)