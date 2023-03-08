import numpy
import random
import connection as cn

ACTIONS = ['left','right','jump']
DIRECTIONS = ['north','east','south','west']
Q_TABLE = None

def save_table(location, table):
    numpy.savetxt(location, table)

def save_best_action(location, table):
    #salva a melhor ação para cada estado a partir da Q_table carregada
    new_table = [numpy.argmax(row) for row in table]
    best_actions = [ACTIONS[index] for index in new_table]
    numpy.savetxt(location, best_actions, fmt='%s')
    print("saved!")
    
def compare_tables(best_actions, desired_best_actions):
    #compara a tabela de best actions salva com a tabela de desired best actions
    try:
        with open(desired_best_actions, 'r') as file:
            lines = file.readlines()
            desired_actions = []
            for line in lines:
                if '/' in line:
                    aux = line.strip().split(' / ')
                    for i in aux:
                        i.replace(' ', '').replace('\n', '')
                    desired_actions+=[aux]
                else:
                    desired_actions.append(line.strip().replace(' ', '').replace('\n', ''))
        try:
            with open(best_actions, 'r') as file:
                lines = file.readlines()
                problem_lines = []
                for i in range(len(lines)):
                    aux = lines[i].replace(' ', '').replace('\n', '')
                    if aux != desired_actions[i] or aux not in desired_actions[i]:
                        problem_lines.append(f"linha: {i} - {lines[i]} - {desired_actions[i]}")
                numpy.savetxt('problem_lines.txt', problem_lines, fmt='%s')
                print("problem lines saved!")
        except FileNotFoundError:
            print("Arquivo best actions não encontrado")
    except FileNotFoundError:
        print("Arquivo desired actions não encontrado")
        
def load_table(location):
    try:
        with open(location) as file:
            table = numpy.loadtxt(file)
    except FileNotFoundError:
        print("Arquivo não encontrado, criando uma tabela vazia")
        table = load_empty_table()

    print("check if the table really was loaded: ")
    print("table[0]: ", table[0])
    return table

def load_empty_table():
    row = [0.0] * len(ACTIONS)
    table = [row] * 96
    table = numpy.array(table)
    return table

#aplcia a função de Q_learning e atualiza na tabela
def update_table(reward, prev_s, prev_a, curr_s, learning_rate = 0.2, discount = 0.9):
    bellman = reward + discount * max(Q_TABLE[curr_s])

    Q_TABLE[prev_s][prev_a] += learning_rate * (bellman - Q_TABLE[prev_s][prev_a])
    return Q_TABLE[prev_s][prev_a]

def extract_state(bit_state):
    plat_mask = 0b1111100
    direction_mask = 0b0000011

    int_state = int(bit_state, 2) #converte de binário pra decimal
    #aplica uma mascara de bits e faz um shift right pra que os bits fiquem no local correto
    platform = (int_state & plat_mask) >> 2
    direction = (int_state & direction_mask)

    print(f"Bits: {bit_state}  \nPlataforma:  {platform} \nDireção: {DIRECTIONS[direction]}\n")

    #acha o estado entre 0-95 a partir da direção e plataforma
    state = platform * 4 + direction
    return state
    
def navigate(socket):
    #reseta o estado atual
    current_state = 0
    
    while True:
        #busca na tabela a melhor ação pra esse estado
        current_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]

        #recebe a recompensa e o estado novos a partir da ação escolhida
        bit_state, reward = cn.get_state_reward(socket, ACTIONS[current_action])
        new_state = extract_state(bit_state)

        current_state = new_state

        print('Ação feita: ' + ACTIONS[current_action])
        print('Novo estado: ' + str(new_state) + '\n')
        
        #para de navegar se input for nao vazio
        stop = input()
        if(stop!=''):
            break


def explore(socket, times, start):
    #reseta o estado atual
    current_state = start * 4 # *4, pois cada plataforma tem 4 estados possíveis e sempre respawna virado pro norte

    for i in range(times):
        print('Passo: ' , i)

        #busca na tabela a melhor ação para o estado atual
        best_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]
        rand_action = random.randint(0, 2)

        #escolhe entre uma ação aleatória ou a melhor ação
        if i % 5 > 0:
            current_action = best_action
        else:
            current_action = rand_action

        #recebe a recompensa e o estado novos a partir da ação escolhida
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
                case 'save best action':
                    if Q_TABLE is None:
                        print('Carregue uma tabela antes')
                    else:
                        save_best_action('best_actions.txt', Q_TABLE)
                case 'compare tables':
                    compare_tables('best_actions.txt', 'desired_best_actions.txt')
                case 'exit':
                    socket.close()
                    break

if __name__ == "__main__":
    main()