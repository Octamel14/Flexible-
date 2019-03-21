# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:10:05 2019
@author: ASUS
"""
import math as m
import random
import numpy as np
import cv2

#random.seed(12)

class Perceptron:
    def __init__(self, NumeroDeNeuronasPorCapa): #Arreglo de enteros ej. [3, 3, 3] son tres capas con tres neuronas
        self.capas=[]
        self.deltas=[]
        self.sigmas=[]
        for i in range(0, len(NumeroDeNeuronasPorCapa)):
            if(i==0):
                self.capas.append(Capa(NumeroDeNeuronasPorCapa[i], NumeroDeNeuronasPorCapa[i]))
            else:
                self.capas.append(Capa(NumeroDeNeuronasPorCapa[i], NumeroDeNeuronasPorCapa[i-1]))
                
    def Activacion(self, inputs):
        
        outputs=[]
        for i in range(0, len(self.capas)):
            outputs=self.capas[i].Activacion(inputs)
            inputs=outputs
        return outputs
    
    def ErrorPorNeurona(self, SalidasReales, SalidasDeseadas):
        error=0
        #print("  ** ", SalidasReales, SalidasDeseadas)
        #print(SalidasReales)
        for i in range (0, len(SalidasReales)):
            #error+=pow((SalidasReales[0][i]-SalidasDeseadas[0][i]), 2)
            error+= pow(pow((SalidasReales[0][i]-SalidasDeseadas[0][i]), 2), 0.5)
        return error
    
    
    def ErrorGeneral(self, inputs, SalidasDeseadas):
        error=0
        for i in range(0, len(inputs)):   
            #print(self.Activacion(inputs[]))
            error+=self.ErrorPorNeurona([self.Activacion(inputs[i])], [SalidasDeseadas[i]])
        print(error)
        return error
    
    def Aprendizaje(self, Entrada, SalidaDeseada, alfa, errorMax):
        error=9999
        while(error>errorMax):
            self.Backpropagation(Entrada, SalidaDeseada, alfa) #
            error=self.ErrorGeneral(Entrada, SalidaDeseada)
            #print(error)
            
    def FijarDeltas(self):
        self.deltas=[]
        
        for i in range(0, len(self.capas)):
            self.deltas.append(np.zeros([len(self.capas[i].neuronas_capa), len(self.capas[i].neuronas_capa[0].pesos)]))
        #print (self.deltas)
     
                #[[[0, 0], [0, 0]], [[0, 0, 0], [0, 0]], [[0], [0, 0, 0]]]
                
        #print (self.deltas)
            #self.deltas=np.zeros((len(self.capas[i].neuronas_capa), len(self.capas[i].neuronas_capa[0].pesos)))
           # self.deltas.append([None*len(self.capas[i].neuronas_capa)]*len(self.capas[i].neurones_capa[0].pesos))
        
        """
        for i in range(0, len(self.capas)):
            self.deltas.append([])
            for j in range(0, len(self.capas[i].neuronas_capa)):
                self.deltas[i].append([])
                for k in range(0, len(self.capas[i].neuronas_capa[0].pesos)):
                    self.deltas[i][j].append(0)
                    
        """
                
    def ActualizarPesos(self, alfa):
        for i in range(0, len(self.capas)):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                for k in range(0, len(self.capas[i].neuronas_capa[j].pesos)):
                    self.capas[i].neuronas_capa[j].pesos[k]-=alfa*self.deltas[i][j, k]
                    
    def ActualizarBias(self, alfa):
        for i in range(0, len(self.capas)):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                self.capas[i].neuronas_capa[j].bias-=alfa*self.sigmas[i][j]
                
    def FijarSigmas(self, SalidaDeseadas):
        self.sigmas=[]
        for i in range(0, len(self.capas)):
            self.sigmas.append([])
            for j in range(0, len(self.capas[i].neuronas_capa)):
                self.sigmas[i].append(0)
            
        for i in range(len(self.capas)-1, -1, -1):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                if(i==len(self.capas)-1):
                    y=self.capas[i].neuronas_capa[j].ultimaactivacion
                    self.sigmas[i][j]=(Neurona(0).Sigmoide(y)-SalidaDeseadas[0][j])*Neurona(0).DerivadaSigmoide(y)
                else:
                    suma=0
                    for k in range(0, len(self.capas[i+1].neuronas_capa)):
                        suma+=self.capas[i+1].neuronas_capa[k].pesos[j]*self.sigmas[i+1][k]
                    
                    self.sigmas[i][j]=Neurona(0).DerivadaSigmoide(self.capas[i].neuronas_capa[j].ultimaactivacion)*suma
                    

       
    def AgregarDelta(self):
        
        for i in range(1, len(self.capas)):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                for k in range(0, len(self.capas[i].neuronas_capa[j].pesos)):
                   # print(i, j, k)
                   
                    self.deltas[i][j, k]+=self.sigmas[i][j]*Neurona(0).Sigmoide(self.capas[i-1].neuronas_capa[k].ultimaactivacion)
                    
                
    def Backpropagation(self, inputs, SalidasDeseadas, alfa):
        self.FijarDeltas()
    
        for i in range(0, len(inputs)):
            self.Activacion(inputs[i])
            self.FijarSigmas([SalidasDeseadas[i]])            
            self.ActualizarBias(alfa)
            self.AgregarDelta()
        self.ActualizarPesos(alfa)
            
        
            
        
            
                  
class Neurona :
    def __init__(self, NumeroEntradas):
        self.pesos=[]
        self.bias=random.random()*10-5
        self.ultimaactivacion=0
        for i in range(0,NumeroEntradas):
            x = random.random()*10-5
            self.pesos.append(x)
            
    def Sigmoide(self, gamma):
        if gamma < 0:
            return 1 - 1 / (1 + m.exp(gamma))
        return 1 / (1 + m.exp(-gamma)) 
    
    def DerivadaSigmoide(self, y):
        z=self.Sigmoide(y)
        return z*(1-z)
    
    def Activacion(self, inputs):
        activacion=self.bias
        for i in range(0, len(self.pesos)):
            activacion+=inputs[i]*self.pesos[i]  
        self.ultimaactivacion=activacion
        return self.Sigmoide(activacion)
    
            
            
class Capa:
    
    def __init__(self, cantidad_neuronas, NumeroEntradas):
        self.neuronas_capa=[]
        self.salida=[]
        for i in range(0, cantidad_neuronas):
            self.neuronas_capa.append(Neurona(NumeroEntradas))
            
    def Activacion(self, inputs):
        salidas=[]
        for i in range(0, len(self.neuronas_capa)):
            salidas.append(self.neuronas_capa[i].Activacion(inputs))
        self.salida=salidas
        return salidas
"""

p=Perceptron([2, 4, 4, 1])
entradas=[]
entradas.append([0, 0])
entradas.append([0, 1])
entradas.append([1, 0])
entradas.append([1, 1])
salidas=[]
salidas.append([1])
salidas.append([0])
salidas.append([0])
salidas.append([1])

p.Aprendizaje(entradas, salidas, 0.3 , 0.01)
"""


p=Perceptron([81, 10, 4])

salidas=[]

for i in range(0, len(crop)):
    _, x1 = cv2.threshold(crop[i], 250, 1, cv2.THRESH_BINARY)
    x2=x1.flatten()
    entradas.append(x2)
    

    

salidas.append([0, 0, 0, 1])
salidas.append([0, 0, 1, 0])
salidas.append([0, 0, 1, 1])
salidas.append([0, 1, 0, 0])
salidas.append([0, 1, 0, 1])
salidas.append([0, 1, 1, 0])
salidas.append([0, 1, 1, 1])
salidas.append([1, 0, 0, 0])
salidas.append([1, 0, 0, 1])
salidas.append([0, 0, 0, 0])
p.Aprendizaje(entradas, salidas, 0.4, 0.01)


