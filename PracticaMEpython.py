"""
Práctica 3: Sistema Musculoesqueletico
Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México
Nombre del alumno: Johana Jazmin De La Torre Gomez
Número de control: 22211751
Correo institucional: L22211751@tectijuana.edu.mx
Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install slycot

#!pip install control
import control as ctrl

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m 
import matplotlib.pyplot as plt 
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx',header=None))
x0,t0,tend,dt,w,h = 0,0,10,1E-3,6,3
N = round((tend-t0)/dt) + 1
t = np.linspace (t0,tend,N)
u = np.zeros(N)
u[round(1/dt):round(2/dt)] = 1 #Impulso

def musculo(Cs,Cp,R,a):
    num = [Cs*R,+1-a]
    den = [R*(Cp+Cs),+1]
    sys = ctrl.tf(num,den)
    return sys

#Funcion de transferencia Control
a,Cs,Cp,R = 0.25,1.10E-6,100E-6,100
syscontrol = musculo(Cs,Cp,R,a)
print (f'Funcion de transferencia de control: {syscontrol}')

#Funcion de transferencia Control
a,Cs,Cp,R = 0.25,1.10E-6,100E-6,10E3
syscaso = musculo(Cs,Cp,R,a)
print (f'Funcion de transferencia de caso: {syscaso}')

#Respuesta en lazo abierto 
_,Fs1 = ctrl.forced_response(syscontrol,t,u,x0)
_,Fs2 = ctrl.forced_response(syscaso,t,u,x0)

fg1 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color=[0.93,0.64,0.35],label= 'F(t): ')
plt.plot(t,Fs1,'-',linewidth=1,color=[0.23,0.67,0.20],label= 'Fs1(t): Control')
plt.plot(t,Fs2,'-',linewidth=1,color=[ 0.90,0.15,0.15],label= 'Fs2(t): Caso')
plt.grid(False) #para que salga cuadricula
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1);plt.yticks(np.arange(-0.1,1.1,0.2))
plt.ylabel('Fi(t) [V]')
plt.xlabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,-0.15), loc='upper center', ncol=4)
plt.show()
fg1.set_size_inches(9,5)
fg1.tight_layout()
fg1.savefig('sistema musculoesqueletico_LA pyhton.png', dpi=600,bbox_inches='tight')
fg1.savefig('sistema musculoesqueletico_LA pyhton.pdf')

def controlador(kI,sys):
    numI = [kI]
    denI = [1,0]
    I = ctrl.tf(numI,denI)
    X = ctrl.series(I,sys)
    sysI = ctrl.feedback(X,1,sign=-1)
    return sysI

I = controlador(28032.4984,syscontrol)

#Respuesta en lazo cerrado 
_,Fs3 = ctrl.forced_response(syscontrol,t,u,x0)


fg2 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color=[0.93,0.64,0.35],label= 'F(t): ')
plt.plot(t,Fs1,'-',linewidth=1,color=[0.23,0.67,0.20],label= 'Fs1(t): Control')
plt.plot(t,Fs2,'-',linewidth=1,color=[ 0.90,0.15,0.15],label= 'Fs2(t): Caso')
plt.plot(t,Fs3,':',linewidth=3,color=[0.68,0.46,0.85],label= 'Fs3(t): Tratamiento')
plt.grid(False) #para que salga cuadricula
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1);plt.yticks(np.arange(-0.1,1.1,0.2))
plt.ylabel('Fi(t) [V]')
plt.xlabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,-0.15), loc='upper center', ncol=4)
plt.show()
fg2.set_size_inches(9,5)
fg2.tight_layout()
fg2.savefig('sistema musculoesqueletico_LC pyhton.png', dpi=600,bbox_inches='tight')
fg2.savefig('sistema musculoesqueletico_LC pyhton.pdf')
