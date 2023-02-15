import numpy
import random
import connection as cn

ACTIONS = ['left','right','jump']
Q_TABLE = None

def save_table(location, table):
    numpy.savetxt(location, table)

def load_table(location):
    try:
        with open(location) as file:
            table = numpy.loadtxt(file)
    except FileNotFoundError:
        table = load_empty_table()

    return table

def load_empty_table():
    row = [0.0] * len(ACTIONS)
    table = [row]*96
    table = numpy.array(table)
    return table

#aplcia a função de Q_learning e atualiza na tabela
def update_table(reward, prev_s, prev_a, curr_s, learning_rate = 0.1, discount = 0.4):
    bellman = reward + discount * max(Q_TABLE[curr_s])

    Q_TABLE[prev_s][prev_a] += learning_rate * (bellman - Q_TABLE[prev_s][prev_a])
    return Q_TABLE[prev_s][prev_a]

def extract_state(bit_state):
    plat_mask = 0b1111100
    direction_mask = 0b0000011

    int_state = int(bit_state, 2)
    #aplica uma mascara de bits e faz um shift right pra que os bits fiquem no local correto
    platform = (int_state & plat_mask) >> 2
    direction = (int_state & direction_mask)

    print(bit_state, platform, direction)
    #acha o estado entre 0-95 a partir da direção e plataforma
    state = platform * 4 + direction
    return state
    
def navigate(socket):
    #reseta o estado atual
    current_state=0
    
    while True:
        #busca na tabela a melhor ação pra esse estado
        current_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]

        #recebe a recompensa e o estado novos a partir da ação escolhida
        bit_state, reward = cn.get_state_reward(socket, ACTIONS[current_action])
        new_state = extract_state(bit_state)

        print(new_state)
        current_state = new_state

        #print_map(current_state)
        print(ACTIONS[current_action])
        
        #para de navegar se input for nao vazio
        stop = input()
        if(stop!=''):
            break


def explore(socket, times, start):
    #reseta o estado atual
    current_state = start * 4

    for i in range(times):
        print(i)
        #busca na tabela a melhor ação pra esse estado e uma ação aleatoria
        best_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]
        rand_action = random.randint(0, 2)

        #escolhe entre uma ação aleatório ou a melhor ação aleatoriamente a partir dos pesos
        if times>50:
            weights = [i % (times/10), (times -1 - i) % (times/10)]
        else:
            weights = [i, times - i]

        current_action = random.choices([best_action, rand_action], weights)[0]

        #recebe a recompensa e o estado novos a aprter da ação escolhida
        bit_state, reward  = cn.get_state_reward(socket, ACTIONS[current_action])
        new_state = extract_state(bit_state)

        update_table(reward, current_state, current_action, new_state)

        current_state = new_state

def main():
    global Q_TABLE
    socket = cn.connect(2037)

    if(socket != 0):
        while True:
            command = input('Envie um comando\n')
            match command:
                case 'save':
                    save_table('resultado.txt', Q_TABLE)
                    print('Tabela salva')
                case 'empty':
                    Q_TABLE = load_empty_table()
                    print('Tabela vazia carregada')
                case 'load':
                    Q_TABLE = load_table('resultado.txt')
                    print('Tabela carregada')
                case 'explore':
                    if Q_TABLE is None:
                        print('Carregue uma tabela antes')
                    else:
                        times = int(input('Quantas vezes ele deve explorar?\n'))
                        start = int(input('Qual a plataforma inicial?\n'))
                        explore(socket, times, start)
                        print('Exploração terminada')
                case 'navigate':
                    if Q_TABLE is None:
                        print('Carregue uma tabela antes')
                    else:
                        print('Aperte Enter para avançar 1 passo, digite algo para terminar\n')
                        navigate(socket)
                case 'exit':
                    socket.close()
                    break

if __name__ == "__main__":
    main()