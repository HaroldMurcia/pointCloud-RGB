#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:15:18 2021

@author: hmurcia
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# Parametros intrinsecos (Camera calibration)
fx = 2454.999259;
fy = 2467.901319;
cx = 2081.630126;
cy = 1096.582888;

# Distortion parameters
k1=0.129679; 
k2=-0.199223;
p1=-0.001564;
p2=0.005949;
k3=0.00000;

pointCloudFile = "cuadros.txt"
referencePointsFile = "cuadros.xls"
imageName = 'cuadros360.jpeg'
numMaxEsquinas = 48

def optimal(X,R):
    # Based on: https://eeweb.engineering.nyu.edu/iselesni/lecture_notes/least_squares/least_squares_SP.pdf
    A = np.dot(X,R.T)
    B = np.linalg.inv(np.dot(X,X.T))
    M = np.dot(B,A)
    
    out = np.dot(M.T,X)
    
    plt.figure()
    plt.plot(out[0,:],out[1,:],'r*',label='box')
    plt.imshow(dst[:,:,::-1])
    plt.legend()
    plt.show()
    
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return M.T


if __name__ == "__main__":
    img = cv2.imread(imageName)
    h, w = img.shape[:2]
    
    intrinsicParameters=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # Define distortion coefficients
    dist = np.array([k1,k2,p1,p2,k3])
    
    # Undistorting the image
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsicParameters,dist,(w,h),1, (w,h))
    dst = cv2.undistort(img, intrinsicParameters, dist, None, newcameramtx)
    
    for j in range(4,numMaxEsquinas+1):
        
    # XYZ points
        df = pd.read_excel (referencePointsFile, nrows = j)
        # Datos de la nube de puntos 
    
        Xn = df["X"]
        Y = df["Y"]
        Z = df["Z"]
    
    # Datos de la imagen corergida
    
        Xp = df["Xp"]
        Yp = df["Yp"]
        X = np.ones([4,len(Xn)])
        X[0,:] = Xn
        X[1,:] = Y
        X[2,:] = Z
        # cam points
        R = np.ones([3,len(Xp)])
        R[0,:] = Xp
        R[1,:] = Yp
        
    #    
        # Optimización para determinar parámetros
        M = optimal(X,R)
    #    print("optimal M",M)
        
        
        '''
        # Lectura de nube de puntos
        '''
        
        nube = np.loadtxt(pointCloudFile)
        
#        indicesXmas = []
    #    Ymas=[]
#        for i in range(0,len(nube)):
#            if(nube[i,2] >= 0.07):
#                indicesXmas.append(i)
#            elif(nube[i,0] >= 0.0):
#                indicesXmas.append(i)
#            elif(nube[i,1] >= 0.05):
#                indicesXmas.append(i)
#            indicesXmas.append(i)
    #        elif  (nube[i,1] >= 0.05):
    #            Ymas.append(i)
                
        pc = np.ones((4, len(nube)))       
        pc[0, :] = nube[:,0]
        pc[1, :] = nube[:,1]
        pc[2, :] = nube[:,2]
        
  
        
        
        '''
        # Calibracion total de la nube de puntos con la imagen
        '''
        n = len(nube)
        calibracionFinal= np.dot(M,pc)
        
   
        # Creacion de nube de puntos final
        
        colorpc = np.zeros((n,6))
        colorpc[:,0] = nube[:,0]
        colorpc[:,1] = nube[:,1]
        colorpc[:,2] = nube[:,2]
        n = len(calibracionFinal.transpose())
   
        '''
        # Asignacion de colores a la nube de puntos basado en la calibracion hecha
        '''
        for i in range(n):
            x= int(calibracionFinal[0,i])
            y= int(calibracionFinal[1,i])
            
            try:
                b,g,r = dst[y,x]
                
            except:
                r = 0
                g = 0
                b = 0
            
            
            colorpc[i , 3]=r
            colorpc[i , 4]=g
            colorpc[i , 5]=b
            
            
        ''' 
            Imagen de nube de puntos sobre la imagen de la cámara
        '''
        plt.figure()
        plt.plot(calibracionFinal[0,:],calibracionFinal[1,:],'r*',label='box')
        plt.plot(R[0,:],R[1,:],'m+',label='box')
        plt.imshow(dst[:,:,::-1])
        plt.legend()
        plt.show()
    
        
        '''
            Calculo del error RMSE
        '''
        errorP = 0
    
         # XYZ points
        df2 = pd.read_excel (referencePointsFile, nrows = numMaxEsquinas)
        # Datos de la nube de puntos 
    
        Xn = df2["X"]
        Y = df2["Y"]
        Z = df2["Z"]
    
    # Datos de la imagen corergida
    
        Xp = df2["Xp"]
        Yp = df2["Yp"]
        X2 = np.ones([4,len(Xn)])
        X2[0,:] = Xn
        X2[1,:] = Y
        X2[2,:] = Z
        # cam points
        R2 = np.ones([3,len(Xp)])
        R2[0,:] = Xp
        R2[1,:] = Yp
        e = np.dot(M,X2)
        for i in range(0,len(e.T)):
            
            eix = (e[0,i] - R2[0,i])*(e[0,i] - R2[0,i])
            eiy = (e[1,i] - R2[1,i])*(e[1,i] - R2[1,i])
            errorP += np.sqrt((eix + eiy)/len(e.T))
            
       
    #    
        print ("Error RMSE con " + str(len(X.T)) + " esquinas")
        print (errorP)
        
        
        
    
    #    plt.figure()
    #    plt.plot(calIzqLateral[0,:],calIzqLateral[1,:],'r*',label='box')
    #    plt.plot(R[0,:],R[1,:],'m+',label='box')
    #    plt.imshow(dstIL)
    #    plt.legend()
    #    plt.show()
    
    
        
        '''
        # Exportación a formato txt
        '''
        
        fileName = str(j) + "esquinas"
        final = open("nubeDePuntosColor_" + fileName + ".txt", "w+")
        dataFile= "%" + "X"+"\t"+ "Y"+"\t"+  "Z" +"\t"+ "R" +"\t" + "G"+"\t"+ "B" + "\n"
        final.write(dataFile)    
    
        for i in range(len(colorpc)):
            x = colorpc[i,0]
            y = colorpc[i,1]
            z = colorpc[i,2]
            r = colorpc[i,3]
            g = colorpc[i,4]
            b = colorpc[i,5]
            data = str(x) + "\t" + str(y) + "\t" + str(z) + "\t" + str(r) + "\t" + str(g) + "\t" + str(b) + "\n"
            final.write(data)
        final.close()
        
    ###        
