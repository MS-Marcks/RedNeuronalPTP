# para realizar el entrenamiento de la red neuronal necesitamos el dependencia MLPClassifier
from sklearn.neural_network import MLPClassifier
from random import choice
import os

# establecer las opciones de juego como en este caso solo hay 3 opciones
options = ["piedra", "tijeras", "papel"]
# se crea una funcion para saber quien de los jugadores gana
def search_winner(p1, p2):
    if p1 == p2:
        result = 0
    elif p1 == "piedra" and p2 == "tijeras":
        result = 1
    elif p1 == "piedra" and p2 == "papel":
        result = 2
    elif p1 == "tijeras" and p2 == "piedra":
        result = 2
    elif p1 == "tijeras" and p2 == "papel":
        result = 1
    elif p1 == "papel" and p2 == "piedra":
        result = 1
    elif p1 == "papel" and p2 == "tijeras":
        result = 2

    return result
# prueba para saber si funciona correctamente la funcion
# print(search_winner("piedra", "piedra"))
# print(search_winner("papel", "piedra"))
# print(search_winner("tijeras", "piedra"))
# print(search_winner("piedra", "tijeras"))
# print(search_winner("papel", "tijeras"))
# print(search_winner("tijeras", "tijeras"))
# print(search_winner("piedra", "papel"))
# print(search_winner("papel", "papel"))
# print(search_winner("tijeras", "papel"))
# print(search_winner("piedra", "piedra"))
# print(search_winner("piedra", "papel"))
# print(search_winner("piedra", "tijeras"))
# print(search_winner("tijeras", "piedra"))
# print(search_winner("tijeras", "papel"))
# print(search_winner("tijeras", "tijeras"))
# print(search_winner("papel", "piedra"))
# print(search_winner("papel", "papel"))
# print(search_winner("papel", "tijeras"))
# test = [
#    ["piedra", "piedra", 0],
#    ["piedra", "tijeras", 1],
#    ["piedra", "papel", 2]
#]
# for partida in test:
#    print("player1: %s player2: %s Winner: %s Validation: %s" % (
#        partida[0], partida[1], search_winner(
#            partida[0], partida[1]), partida[2]
#    ))

# definir una funcion para seleccionar un numero alzar con la cantidad de elementos de la opciones de juego
def get_choice():
    return choice(options)

# realizamos una prueba con opciones al alzar para comprobar si funciona la funcion 
#for i in range(10):
#    player1 = get_choice()
#    player2 = get_choice()
#    print("player1: %s player2: %s Winner: %s" % (
#        player1, player2, search_winner(player1, player2)
#    ))

# como en la red neuronal trabaja con vectores o numeros hay que crear una especie de codificacion 
# para que la red neuronal comprenda que cada valor signifique algo
def str_to_list(option):
    if option == "piedra":
        res = [1, 0, 0]
    elif option == "tijeras":
        res = [0, 1, 0]
    else:
        res = [0, 0, 1]
    return res

# se crea una lista con opciones de juego para establecer el aprendizaje
# data_x es el jugador 1
# data_y es el jugador 2
data_x = list(map(str_to_list, ["piedra", "tijeras", "papel"]))
data_y = list(map(str_to_list, ["papel", "piedra", "tijeras"]))

print(data_x)
print(data_y)

# se inicializar la variable que tomara como valor la red neuronal
clf = MLPClassifier(verbose=False, warm_start=True)

# en este caso solo se selecciona la opcion uno para que aprenda la red neuronal 
model = clf.fit([data_x[0]], [data_y[0]])
print(model)

# se crea una funcion de aprendizaje de la red neuronal

def play_and_learn(iters=1000, debug=False):

    # se crea una lista para establecer cuanta veces gano la red o cuantas veces perdio
    score = {"win": 0, "loose": 0}

    # se crean dos listas para guardar las opciones que selecciono cada jugador
    # data_x jugador 1
    # data_y jugador 2
    data_x = []
    data_y = []

    # el for servira para realizar las iteraciones o ciclos de aprendizaje de la red neuronal
    for i in range(iters):
        # el jugador 1 se estabecera con opciones random 
        player1 = get_choice()
        
        # juego vamos a obtener la probabilidad que tenga la red neuronal para elegir entre las opciones
        predict = model.predict_proba([str_to_list(player1)])[0]
        
        # el primer es saber si esta en un 95% seguro que tiene que elegir piedra para ganar 
        if predict[0] >= 0.95:
            player2 = options[0]
        # el segundo es saber si esta en un 95% seguro que tiene que elegir tijeras para ganar 
        elif predict[1] >= 0.95:
            player2 = options[1]
        # el tercero es saber si esta en un 95% seguro que tiene que elegir papel para ganar 
        elif predict[2] >= 0.95:
            player2 = options[2]
        # el cuarto por si no esta seguro de cual elegir entonces que elija al alzar una opcion
        else:
            player2 = get_choice()
        # impirmir un mensaje para saber los valores obtendios en la iteracion anterior
        if debug == True:
            print("player1: %s player2 (Modelo): %s ---> %s" % (player1, predict, player2))
        # guardamos en una variable quien gano la jugada
        winner = search_winner(player1, player2)
        # imprimimos quien gano la jugada
        if debug == True:
            print("Comprobamos p1 VS p2: %s" % winner)
        # si gana entonces se guarda la partida en las lista para posteriormente aprender con que se gana
        if winner == 2:
            data_x.append(str_to_list(player1))
            data_y.append(str_to_list(player2))
            score["win"] += 1
        # se lo contrario no se hace nada
        else:
            score["loose"] += 1
    # retorna los valores importantes para que la red neuronal aprenda
    return score, data_x, data_y

# realizamos el primer entremamiento de prueba para ver si funciona 
# score,data_x,data_y = play_and_learn(1,debug=False)
# print(data_x)
# print(data_y)
# print("score: %s %s %%" % (score, (score["win"]*100/(score["win"]+score["loose"]))))
# si en el entremamiento la red neuronal gano partidas entonces se entrena el modelo de nuevo 
# para que aprenda esas nuevos conocimientos
# if len(data_x):
#    model = model.partial_fit(data_x,data_y)

# ahora realizamos el verdadero entrenamiento de de la red neuronal
# establecemos una variables para saber cuantas iteraciones de aprendizaje de realizo
i = 0
# esteblecmos una lista para saber el historial de prediciones que tuvo la red neuronal
historic_pct = []

# se estable un while en true para que aprenda 
while True:
    i += 1
    # realiza el aprendizaje con 1000 iteraciones para el aprendizaje
    score, data_x, data_y = play_and_learn(1000, debug=False)
    # se guarda el historia de la preidcion
    ptc = (score["win"]*100/(score["win"]+score["loose"]))
    historic_pct.append(ptc)
    # se imprime los datos obtendios
    print("Iter: %s - score: %s %s %%" % (i, score, ptc))
    # y si hubo partidas ganas de parte de la red neuronal entonces se realizo el aprendizaje de la red
    if len(data_x):
        model = model.partial_fit(data_x, data_y)

    # el entramiento se parara cuando la red neuronal aprenda a ganar en las ultimas 9 iteracciones de aprendizaje gane todas
    if sum(historic_pct[-9:]) == 900:
        break

# se guardan en un documento para expresar en una grafica como fue la curva de aprendizaje de la red neuronal
file = open("linea de aprendizaje.csv", "w")
for i in range(len(historic_pct)):
    file.write("%s;%s\n" % (i,historic_pct[i]))
file.close()