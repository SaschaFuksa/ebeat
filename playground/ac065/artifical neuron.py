#Video 3 von Valerio Velardo, Deep Learning (for Audio) with Python, damit man das mit den weights und inputs versteht
#if name etc. damit das Skript runt
import math
def sigmoid (x):
    y = 1.0 / (1+ math.exp(-x))
    return y
def activate(inputs, weights):
    #perform net input (also inuts und weights multiplizieren und miteinader addieren)
    h = 0 #predifine h, start von 0 un dann loopt es in der for schleife durch, zip funktion
    for x, w in zip(inputs, weights):
        h += x*w
    # perform activation
    return sigmoid(h)
if __name__== "__main__":
    inputs = [.5, .3, .2]
    weights = [ .4, .7, .2]
    #inputs und weights werden als Liste repräsentiert ( beispiel aus Video 3, screenshot in Dokument LSTM Ordner Technology Lab)
    output = activate (inputs, weights)
    print(output)