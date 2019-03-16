# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:10:05 2019

@author: ASUS
"""
import math as m
import random
import numpy as np


class Perceptron:
    def __init__(self, NumeroDeNeuronasPorCapa): #Arreglo de enteros ej. [3, 3, 3] son tres capas con tres neuronas
        self.capas=[]
        self.deltas=[]
        self.sigmas=[]
        for i in range(0, len(NumeroDeNeuronasPorCapa)):
            if(i==0):
                self.capas.append(Capa(NumeroDeNeuronasPorCapa[i], NumeroDeNeuronasPorCapa[i]))
            else:
                self.capas.append(Capa(NumeroDeNeuronasPorCapa[i-1], NumeroDeNeuronasPorCapa[i]))
                
    def Activacion(self, inputs):
        
        outputs=0
        for i in range(0, len(self.capas)):
            outputs=self.capas[i].Activacion(inputs)
            inputs=outputs
            #print(outputs)
        #print("Longitud ", len(outputs), "Valor ", outputs)
        return outputs[len(outputs)-1]
    
    def ErrorPorNeurona(self, SalidasReales, SalidasDeseadas):
        error=0
        for i in range (0, len(SalidasReales)):
            #print("La salida es ", SalidasDeseadas[i])
            #print("La salida esperada es ", SalidasReales[i])
            error+=  m.pow((SalidasReales[i]-SalidasDeseadas[i]), 2)
        print(SalidasReales)
        return error
    
    
    def ErrorGeneral(self, inputs, SalidasDeseadas):
        error=0
        for i in range(0, len(inputs)):
            
            error+=self.ErrorPorNeurona([self.Activacion(inputs[i])], [SalidasDeseadas[i]])
        return error
    
    def Aprendizaje(self, Entrada, SalidaDeseada, alfa, errorMax):
        error=99999
        while(error>errorMax):
            self.Backpropagation(Entrada, SalidaDeseada, alfa) #Posible Correcion
            error=self.ErrorGeneral(Entrada, SalidaDeseada)
            #print(error)
            
    def FijarDeltas(self):
        
        
        for i in range(0, len(self.capas)):
            self.deltas.append([])
            for j in range(0, len(self.capas[i].neuronas_capa)):
                self.deltas[i].append([])
                for k in range(0, len(self.capas[i].neuronas_capa[0].pesos)):
                    self.deltas[i][j].append(0)
     
                #[[[0, 0], [0, 0]], [[0, 0, 0], [0, 0]], [[0], [0, 0, 0]]]
                
            
            #self.deltas=np.zeros((len(self.capas[i].neuronas_capa), len(self.capas[i].neuronas_capa[0].pesos)))
           # self.deltas.append([None*len(self.capas[i].neuronas_capa)]*len(self.capas[i].neurones_capa[0].pesos))
        
        """
        for i in range(0, len(self.capas)):
            self.deltas.append(len(self.capas[i].neuronas_capa), len(self.capas[i].neuronas_capa[0].pesos))
            for j in range(0, len(self.capas.neuronas_capa)):
                for k in range(0, len(self.capas[i].neuronas_capa[0].pesos)):
                    self.deltas[i][j, k]=0
                    
        """
                
    def FijarPesos(self, alfa):
        for i in range(0, len(self.capas)):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                for k in range(0, len(self.capas[i].neuronas_capa[j].pesos)):
                    self.capas[i].neuronas_capa[j].pesos[k]-=alfa*self.deltas[i][j][k]
                    
    def FijarBias(self, alfa):
        for i in range(0, len(self.capas)):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                self.capas[i].neuronas_capa[j].bias-=alfa*self.sigmas[i][j]
                
    def FijarSigmas(self, SalidaDeseadas):
        for i in range(0, len(self.capas)):
            self.sigmas.append([])
            for j in range(0, len(self.capas[i].neuronas_capa)):
                self.sigmas[i].append(0)
            
        for i in range(len(self.capas)-1, 0, -1):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                if(i==len(self.capas)-1):
                    
                    y=self.capas[i].neuronas_capa[j].ultimaactivacion
                    self.sigmas[i][j]=(Neurona(0).Sigmoide(y)-SalidaDeseadas[j])*Neurona(0).DerivadaSigmoide(y)
                else:
                    suma=0
                    for k in range(len(self.capas[i+1].neuronas_capa)):
                        suma+=self.capas[i+1].neuronas_capa[k].pesos[j]*self.sigmas[i+1][k]
                    
                    self.sigmas[i][j]=Neurona(0).DerivadaSigmoide(self.capas[i].neuronas_capa[j].ultimaactivacion)*suma
                    
    def ActualizarPeso(self, alfa):
        return 0
    
    def ActualizarBias(self, alfa):
        return 0
                    
                
                
       
    def AgregarDelta(self):
        
        for i in range(1, len(self.capas)):
            for j in range(0, len(self.capas[i].neuronas_capa)):
                for k in range(0, len(self.capas[i].neuronas_capa[j].pesos)):
                   
                    self.deltas[i][j][k]+=self.sigmas[i][j]*Neurona(0).Sigmoide(self.capas[i-1].neuronas_capa[k].ultimaactivacion)
                    
                
    def Backpropagation(self, inputs, SalidasDeseadas, alfa):
        self.FijarDeltas()
    
        for i in range(0, len(inputs)):
            self.Activacion(inputs[i])
            self.FijarSigmas([SalidasDeseadas[i]])            
            self.FijarBias(alfa)
            self.AgregarDelta()
        self.FijarPesos(alfa)
            
        
            
        
            
                  
class Neurona :
    def __init__(self, NumeroEntradas):
        self.pesos=[]
        self.bias=random.random()
        self.ultimaactivacion=0
        self.activacion=self.bias
        for i in range(0,NumeroEntradas):
            x = random.random()
            self.pesos.append(x)
    def Sigmoide(self, y):
        return (1/(1+m.exp(-y)))
    
    def DerivadaSigmoide(self, y):
        z=self.Sigmoide(y)
        return z*(1-z)
    
    def Activacion(self, inputs):
        self.activacion=self.bias
        for i in range(0, len(self.pesos)):
          
            self.activacion+=inputs[i]*self.pesos[i]
        self.ultimaactivacion=self.activacion
        return self.Sigmoide(self.activacion)
            
            
class Capa:
    
    def __init__(self, NumeroEntradas, cantidad_neuronas):
        self.neuronas_capa=[]
        self.salidas=[]
        for i in range(0, cantidad_neuronas):
            self.neuronas_capa.append(Neurona(NumeroEntradas))
            
    def Activacion(self, inputs):
        for i in range(0, len(self.neuronas_capa)):
            self.salidas.append(self.neuronas_capa[i].Activacion(inputs))
        return self.salidas

entradas=[]
entradas.append([0, 0])
entradas.append([0, 1])
entradas.append([1, 0])
entradas.append([1, 1])
salidas=[]
salidas.append(1)
salidas.append(0)
salidas.append(0)
salidas.append(1)

p=Perceptron([2, 3, 1])
p.Aprendizaje(entradas, salidas, 0.3, 0.1)
        