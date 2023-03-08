import numpy
import random
import connection as cn

ACTIONS = ['left','right','jump']
DIRECTIONS = ['north','east','south','west']
Q_TABLE = None

def save_table(location, table):
    numpy.savetxt(location, table)

def save_best_action(location, table):
    """
    Salva a melhor ação para cada estado a partir da Q_table carregada
    """
    new_table = [numpy.argmax(row) for row in table]
    best_actions = [ACTIONS[index] for index in new_table]
    numpy.savetxt(location, best_actions, fmt='%s')

def compare_tables(best_actions, desired_best_actions):
    """
    Compara a tabela de best actions salva com a tabela de desired best actions
    """
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
                        platform = int(i/4)
                        direction = DIRECTIONS[i%4]
                        problem_lines.append(f"linha do arquivo: {i+1} (platform {platform} {direction}) - {lines[i]} - should be: {desired_actions[i]}")
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

def update_table(reward: int, prev_s: int, prev_a: int, curr_s: int, learning_rate: float = 0.2, discount: float = 0.9):
    """
    Atualiza a a tabela usando a equação de atualização

    Parâmetros:
    reward (int): a recompesna recebida
    prev_s (int): o estado anterior
    prev_a (int): a ação realizada
    curr_s (int): o estado atual
    learning_rate (int): o valor do learning rate
    discount (int): o valor do desconto
    """
    bellman = reward + discount * max(Q_TABLE[curr_s])

    #equação de atualização
    Q_TABLE[prev_s][prev_a] += learning_rate * (bellman - Q_TABLE[prev_s][prev_a])
    return Q_TABLE[prev_s][prev_a]

def extract_state(bit_state: str):
    """
    Extrai o estado a partir do valor binário

    Parâmetros:
    bit_state (str): valor binário que representa a plataforma e direção
    
    Retorna:
    state (int): estado extraido
    """
    plat_mask = 0b1111100
    direction_mask = 0b0000011

    int_state = int(bit_state, 2) #converte de binário pra decimal
    #aplica uma mascara de bits e faz um shift right pra que os bits fiquem no local correto
    platform = (int_state & plat_mask) >> 2
    direction = (int_state & direction_mask)

    #acha o estado entre 0-95 a partir da direção e plataforma
    state = platform * 4 + direction
    return state
    
def navigate(socket, times: int, start: int):
    """
    Navega pelo mapa usando a melhor ação de cada estado

    Parâmetros:
    socket: socket para a conexão
    times (int): quantidade de passos que irão ser feitos
    start (int): plataforma inicial
    """
    #reseta o estado atual
    current_state = start * 4
    total_reward = 0

    for i in range(times):
        #busca na tabela a melhor ação pra esse estado
        current_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]

        #recebe a recompensa e o estado novos a partir da ação escolhida
        bit_state, reward = cn.get_state_reward(socket, ACTIONS[current_action])
        new_state = extract_state(bit_state)

        total_reward += reward
        current_state = new_state

        print('Ação feita: ' + ACTIONS[current_action])
        print(f'Novo estado: {new_state}')
        print(f'Recompensa média: {round(total_reward/(i+1), 3)} \n')


def explore(socket, times: int, start: int):
    """
    Explora o mapa e atualiza a tabela com base na recompensa recebida

    Parâmetros:
    socket: socket para a conexão
    times (int): quantidade de passos que irão ser feitos
    start (int): plataforma inicial
    """
    #reseta o estado atual
    current_state = start * 4 # *4, pois cada plataforma tem 4 estados possíveis e sempre respawna virado pro norte

    for i in range(times):
        print('Passo: ' , i)

        #busca na tabela a melhor ação para o estado atual
        best_action = numpy.where(Q_TABLE[current_state] == max(Q_TABLE[current_state]))[0][0]
        rand_action = random.randint(0, 2)

        # ---------------- # ! Lógicas de exploration x exploitation (ambas foram usadas) # ---------------------
        # ? escolhe entre uma ação aleatório ou a melhor ação aleatoriamente a partir dos pesos
        # if times>50:
        # weights = [i % (times/10), (times -1 - i) % (times/10)]
        # else:
        # weights = [i, times - i]
        # current_action = random.choices([best_action, rand_action], weights)[0]

        # ? escolhe entre uma ação aleatória ou a melhor ação
        if i % 5 > 1:
            current_action = best_action
        else:
            current_action = rand_action
        # ---------------- # ! Lógicas de exploration x exploitation (ambas foram usadas) # ---------------------

        #recebe a recompensa e o estado novos a partir da ação escolhida
        bit_state, reward  = cn.get_state_reward(socket, ACTIONS[current_action])

        new_state = extract_state(bit_state)

        update_table(reward, current_state, current_action, new_state)

        current_state = new_state

def main():
    global Q_TABLE

    socket = cn.connect(2037)
    
    if(socket != 0):
        print('\nDigite "help" para ver os comandos')
        while True:
            command = input('\nInsira um comando\n')
            match command:
                case 'save':
                    save_table('resultado.txt', Q_TABLE)
                    save_best_action('best_actions.txt', Q_TABLE)
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
                        try:
                            times = int(input('Quantas vezes ele deve explorar?\n'))
                            start = int(input('Qual a plataforma inicial?\n'))
                        except ValueError:
                            print('Valor inválido')
                        else:
                            explore(socket, times, start)
                            print('Exploração terminada')
                case 'navigate':
                    if Q_TABLE is None:
                        print('Carregue uma Q_table antes')
                    else:
                        try:
                            times = int(input('Quantas vezes ele deve percorrer?\n'))
                            start = int(input('Qual a plataforma inicial?\n'))
                        except ValueError:
                            print('Valor inválido')
                        else:
                            navigate(socket, times, start)
                case 'compare tables':
                    compare_tables('best_actions.txt', 'desired_best_actions.txt')

                case 'help':
                    print('\nsave: Salva a Q_table e suas melhores ações para cada estado nos arquivos de texto\n')
                    print('empty: Carrega uma Q_table vazia\n')
                    print('load: Carrega a Q_table do arquivo de texto\n')
                    print('explore: Explora o mapa e atualiza a Q_table a partir das recompensas\n')
                    print('navigate: Navega o mapa usando a melhor ação de cada estado\n')
                    print('compare tables: Compara as melhores ações salvas com as melhores ações desejadas\n')
                    print('exit: Termina o programa')
                case 'exit':
                    socket.close()
                    break

if __name__ == "__main__":
    main()
